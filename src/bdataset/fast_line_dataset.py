# Released under MIT license
# Copyright (c) 2024 zhifeng.ding (vivo)
import math
import mmap
from tqdm import tqdm
import contextlib
from pathlib import Path
import os
import pickle
import random
from timer import timer, Timer
import numpy as np
from loguru import logger
from .image_bucket import ImageBuckets


class InjectDataset:
    def __init__(self, datas=[]):
        self.datas = datas
    
    def __len__(self):
        return len(self.datas)
    
    def clean(self):
        self.datas = []
    
    def append(self, data):
        self.datas.append(data)
    
    def transforms(self, idx):
        raise NotImplementedError("return a tensor or dict/list of tensor")
    
    def __getitem__(self, idx):
        # return a single data
        if idx > len(self.datas):
            raise IndexError
        data = None
        while True:
            try:
                data = self.transforms(idx)
                break
            except:
                logger.warning(f'error data: {idx}')
                idx = random.randint(0, self.__len__()-1) 

        return data 


class InjectBucketDataset:
    def __init__(self, buckets: ImageBuckets):
        self.bucket = buckets
        self.datas = []
    
    def __len__(self):
        return len(self.bucket)
    
    def clean(self):
        self.datas = []
        self.bucket.clean()
    
    def data2node(self, line_data):
        raise NotImplementedError("return a BuckNode")
    
    def append(self, line_data):
        flag = self.bucket.inject(self.data2node(line_data))
        if flag != -1:
            self.datas.append(line_data)
    
    def transforms(self, idx, resolution):
        raise NotImplementedError("return a tensor or dict/list of tensor")
    
    def totensor(self, datas):
        raise NotImplementedError("batch a list datas and return a tensor or dict/list of tensor")

    def shuffle(self, epoch):
        # shuffle the bucket
        self.bucket.shuffle(epoch)

    def make(self, batch_size, epoch_seed=0):
        # make a bucket dataset, please call this function before training
        self.bucket.make(batch_size, epoch_seed=epoch_seed)

    def __getitem__(self, idx):
        # return a batch data
        datas = []
        bidxs, resolution, bucket = self.bucket[idx]
        for bdx in bidxs:
            while True:
                try:
                    data = self.transforms(bdx, resolution)
                    datas.append(data)
                    break
                except:
                    logger.warning(f'error data: {bdx}')
                    bdx = random.choice(bucket)
        return self.totensor(datas)


class FastLineReader:
    def __init__(self, file_path, index_path=None, parse=None, skip_head:bool=False):
        self.file_path = Path(file_path)
        self.fp = open(self.file_path, 'r')
        self.mmap = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ) 

        if not isinstance(index_path, (tuple, list)):

            if index_path is None:
                self.index_path = Path(self.file_path.as_posix() + '.index')
                if not self.index_path.exists():
                    self.index_path = Path(self.file_path.name + '.index')
            else:
                self.index_path = Path(index_path)
                if self.index_path.suffix != '.index':
                    raise RuntimeError(f"index_path suffix must be .index, got {self.index_path.suffix}")
            
            if not self.index_path.exists():
                self.build(skip_head=skip_head)

            self.jdict = pickle.load(self.index_path.open('rb')) 
        else:
            self.index_path = index_path[0]
            self.jdict = index_path[1]
        
        if self.jdict['size'] != os.path.getsize(self.file_path):
            self.build(skip_head=skip_head)
            self.jdict = pickle.load(self.index_path.open('rb'))

        self.__parse = parse if parse else lambda x: x
    
    @classmethod
    def from_index(cls, index_path, parse=None):
        index_path = Path(index_path)
        jdict = pickle.load(index_path.open('rb'))
        file_path = jdict['source']
        skip_head = jdict.get('skip_head', False)
        return cls(file_path, (index_path, jdict), parse=parse, skip_head=skip_head)


    @timer('build index')
    def build(self, skip_head=False):
        with open(self.index_path, 'wb') as f:
            jdict = {"source": self.file_path.absolute(), "size": os.path.getsize(self.file_path), "skip_head": skip_head, "fpos": []}
            with contextlib.closing(mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)) as m:
                b = m.tell()
                pbar = tqdm(total=m.size(), desc=f'{self.index_path.stem} indexing', colour='green')
                while b < m.size():
                    _ = m.readline()
                    e = m.tell()
                    diff = e - b
                    if diff > 2:
                        jdict['fpos'].append(b)
                    pbar.update(diff)
                    b = e
                pbar.close()
            if skip_head:
                del jdict['fpos'][0]
            pickle.dump(jdict, f)
    
    def __del__(self):
        self.mmap.close()
        self.fp.close()
    
    def __len__(self):
        return len(self.jdict['fpos'])
    
    @timer("shuffle") 
    def shuffle(self):
        random.shuffle(self.jdict['fpos'])
    
    def __getitem__(self, idx):
        fpos = self.jdict['fpos'][idx]
        self.mmap.seek(fpos)
        return self.__parse(self.mmap.readline().strip())


class FastLineDataset:
    def __init__(self, readers: FastLineReader):
        self.readers = readers
        sizes = np.array([0] + [len(reader) for reader in self.readers])
        self.sizes = np.cumsum(sizes)
    
    @classmethod
    def from_index(cls, index_files, parse=None):
        return cls([FastLineReader.from_index(index_file, parse=parse) for index_file in index_files])

    def __len__(self):
        return self.sizes[-1]
    
    def transforms(self, idx):
        return self.read_line(idx)
    
    def read_line(self, idx):
        # 读取文件某一行，可重载；如果想过滤该行，将返回值置为None
        jdx = (self.sizes[1:] <= idx).sum()
        kdx = idx - self.sizes[jdx]
        return self.readers[jdx][kdx]
    
    def __getitem__(self, idx):
        data = None
        while True:
            try:
                data = self.transforms(idx)
                break
            except:
                logger.warning(f'error data: {idx}')
                idx = random.randint(0, self.__len__()-1) 

        return data
    
    def inject_count(self, chunk_size):
        n = self.sizes[-1]
        return math.ceil(n / chunk_size)
    
    def inject(self, inject_module, chunk_size=10_000_000, shuffle=True):
        n = self.sizes[-1]
        idxs = np.arange(n)
        if shuffle:
            with Timer('shuffle'):
                np.random.shuffle(idxs)

        dataset = inject_module
        nck = self.inject_count(chunk_size) 
        for i in range(nck):
            b = i * chunk_size
            e = min(b + chunk_size, n)
            dataset.clean()
            sub_idx = np.sort(idxs[b:e])
            for j in tqdm(range(len(sub_idx)), colour='green', desc=f"injecting {i}/{nck}"):
                dataset.append(self.read_line(sub_idx[j]))
            yield dataset