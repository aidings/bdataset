# bdataset
> a big dataset for network training

## install
`pip install git+https://github.com/aidings/bdataset.git`

## example
```python
from bdataset import Jsonl

# 1.转换jsonl文件为hdf5文件
jdict = Jsonl(header=['file_path', 'caption'])
# note: 转hdf5只支持原子类型的转换，如果失败，使用pickle.dumps转化为二进制
# jdict['caption'] = True 表示使用caption的数据使用pickle.dumps转换为二进制类型
jdict.export('sample_1', json_path='sample.jsonl')

# 2.读取hdf5文件,分chunk构建dataset
ck = ChunkDatatset('./sample_1', chunk_size=15, shuffle=True)
print(len(ck))
for i in range(len(ck)):
    print('='*20)
    ck.inject(i)
    for data in ck:
        print(data)
```