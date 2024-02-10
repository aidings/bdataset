import vaex
import math
from pathlib import Path
from timer import Timer

class HDF5:
    def __init__(self, data_dict=None, prefix='bdata'):
        self.data_dict = data_dict if data_dict else {}
        self.t = Timer(prefix)

    @staticmethod
    def save2hdf5(data_dict, root_path='./sample', name='bdata', npart=1, progress=True):
        Path(root_path).mkdir(parents=True, exist_ok=True)

        df = vaex.from_dict(data_dict) 
        ndf = len(df)
        if npart <= 1:
            df.export(f"{root_path}/{name}.hdf5", progress=progress, parallel=True, virtual=True, shuffle=False)
        else:
            base = math.ceil(len(df) / npart)
            for i in range(npart):
                start = i * base
                end = min((i+1) * base, ndf)
                df[start:end].export(f"{root_path}/{name}_{i}.hdf5", progress=progress, parallel=True, virtual=True, shuffle=False)
    
    def _parse(self, *args, **kwargs):
        return self.data_dict
    
    def export(self, root_path='./sample', name='bdata', npart=1, progress=True, *args, **kwargs):
        self.t.start()
        self.data_dict = self._parse(*args, **kwargs)
        HDF5.save2hdf5(self.data_dict, root_path=root_path, name=name, npart=npart, progress=progress)
