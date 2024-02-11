import os
import math
import vaex
from pathlib import Path


class ChunkDatatset:
    def __init__(self, root_path, chunk_size=None, shuffle=False):
        path = Path(root_path)
        if path.suffix == '.hdf5':
            self.df = vaex.open(root_path)
        else:
            file_paths = list(path.glob('*.hdf5'))
            file_paths = sorted(file_paths, key=lambda x: os.path.getatime(x))
            self.df = vaex.open(file_paths)
        self.df = self._filter()
        self.chunk_size = chunk_size if chunk_size else len(self.df)
        self.datas = []
        if shuffle:
            self.shuffle()
    
    def shuffle(self):
        self.df = self.df.shuffle()
    
    def _filter(self):
        return self.df
    
    def __len__(self):
        return math.ceil(len(self.df) / self.chunk_size)

    def inject(self, idx):
        start = idx * self.chunk_size
        end = min(start + self.chunk_size, len(self.df))
        self.datas = self.df[start:end].to_arrays(array_type='list')    
    
    def __getitem__(self, idx):
        return [val[idx] for val in self.datas]