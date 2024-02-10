import os
import math
import vaex
from pathlib import Path



class ChunkDatatset:
    def __init__(self, root_path, chunk_size=None):
        path = Path(root_path)
        if path.suffix == '.hdf5':
            self.df = vaex.open(root_path)
        else:
            file_paths = list(path.glob('*.hdf5'))
            file_paths = sorted(file_paths, key=lambda x: os.path.gettime(x))
            self.df = vaex.open(file_paths)
        self.chunk_size = chunk_size if chunk_size else len(self.df)
    
    def shuffle(self):
        self.df = self.df.shuffle()
    
    def __len__(self):
        return math.ceil(len(self.df) / self.chunk_size)

    def __iter__(self):
        return self
    
    def __next__(self):
        ndf = len(self.df)
        for i in range(0, ndf, self.chunk_size):
            end = min(i+self.chunk_size, ndf)
            yield self.df[i:end].to_arrays(array_type='list')
        raise StopIteration

        
    @staticmethod
    def save2hdf5(data_dict, root_path='./sample', name='bdata', npart=1):
        Path(root_path).mkdir(parents=True, exist_ok=True)

        df = vaex.from_dict(data_dict) 
        ndf = len(df)
        if npart <= 1:
            df.export(f"{root_path}/{name}.hdf5", progress=True, parallel=True, virtual=True, shuffle=False)
        else:
            base = math.ceil(len(df) / npart)
            for i in range(npart):
                start = i * base
                end = min((i+1) * base, ndf)
                df[start:end].export(f"{root_path}/{name}_{i}.hdf5", progress=True, parallel=True, virtual=True, shuffle=False)
