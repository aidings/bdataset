import jsonlines
from typing import List
from .data_dict import HDF5, Header


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
    
    def decode(self, __name: str, __value):
        return self.header[__name]['decode'](self.data_dict[__value])