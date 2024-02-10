import vaex
import math
from pathlib import Path

class HDF5:
    def __init__(self, data_dict=None):
        self.data_dict = data_dict if data_dict else {}

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
    
    def _parse(self, *args, **kwargs):
        return self.data_dict
    
    def export(self, root_path='./sample', name='bdata', npart=1, *args, **kwargs):
        data_dict = self._parse(*args, **kwargs)
        HDF5.save2hdf5(data_dict, root_path=root_path, name=name, npart=npart)
