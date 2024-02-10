import jsonlines
from typing import List, Callable
import pickle
from .data_dict import HDF5


class Header:
    def __init__(self, header:List=[]):
        self.header = {}
        for key in header:
            self[key] = None

    @staticmethod 
    def __identify(__value):
        return __value

    def __setitem__(self, __name: str, __value: Callable) -> None:
        self.header[__name] = {'encode': None, 'decode': None}
        self.header[__name]['encode'] = Header.__identify if __value is None else pickle.dumps
        self.header[__name]['decode'] = Header.__identify if __value is None else pickle.loads
    
    def __getitem__(self, __name: str):
        raise NotImplementedError
    
    def encode(self, __name: str, __value):
        return self.header[__name]['encode'](__value)
    
    def decode(self, __name: str, __value):
        return self.header[__name]['decode'](__value)
    
    def create(self):
        return dict([(key, []) for key in self.header.keys()]) 


class Jsonl(Header, HDF5):
    def __init__(self, header:List=[], data_dict=None, prefix='bdata'):
        Header.__init__(self, header)
        HDF5.__init__(self, data_dict, prefix) 

    def _parse(self, *args, **kwargs):
        if not (json_path := kwargs.get('json_path', None)): 
            json_path = args[0]
        data_dict = self.create()

        idx = 1 
        with jsonlines.open(json_path) as reader:
            for item in reader.iter(type=dict, skip_invalid=True):
                for key in self.header.keys():
                    data_dict[key].append(self.encode(key, item.get(key, None)))
                print(f'timer:{self.t.running()} file:{idx:08d}', end='\r')
                idx += 1
        print(f'timer:{self.t.running()} file:{idx:08d}')
 
        return data_dict