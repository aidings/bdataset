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


class InjectDataset:
    def __init__(self):
        self.datas = []
    
    def __len__(self):
        return len(self.datas)
    
    def clean(self):
        self.datas = []
    
    def append(self, data):
        self.datas.append(data)
    
    def transforms(self, idx):
        return self.datas[idx]
    
    def __getitem__(self, idx):
        return self.transforms(idx)


class FastLineReader:
    def __init__(self, file_path, index_path=None, parse=None, skip_head:bool=False):
        self.file_path = Path(file_path)
        self.fp = open(self.file_path, 'r') 

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

        self.mmap = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
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
            jdict = {"source": self.file_path, "size": os.path.getsize(self.file_path), "skip_head": skip_head, "fpos": []}
            with contextlib.closing(mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)) as m:
                b = m.tell()
                pbar = tqdm(total=m.size(), desc='indexing', colour='green')
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
    def __init__(self, readers):
        self.readers = readers
        sizes = np.array([0] + [len(reader) for reader in self.readers])
        self.sizes = np.cumsum(sizes)

    def __len__(self):
        return self.sizes[-1]
    
    def transforms(self, idx):
        return self.read_line(idx)
    
    def read_line(self, idx):
        jdx = (self.sizes[1:] <= idx).sum()
        kdx = idx - self.sizes[jdx]
        return self.readers[jdx][kdx]
    
    def __getitem__(self, idx):
        return self.transforms(idx)
    
    def inject(self, subset_module:InjectDataset, chunk_size=10_000_000, shuffle=True):
        idxs = np.arange(self.sizes[-1])
        if shuffle:
            with Timer('shuffle'):
                np.random.shuffle(idxs)

        dataset = subset_module
        nck = math.ceil(len(idxs) / chunk_size)
        for i in range(nck):
            b = i * chunk_size
            e = min(b + chunk_size, len(idxs))
            dataset.clean()
            for j in tqdm(range(b, e), colour='green', desc=f"inject:{i:02d}/{nck}"):
                dataset.append(self.read_line(j))
            yield dataset