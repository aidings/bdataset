from bdataset import ChunkDatatset, Jsonl


if __name__ == '__main__':
    jdict = Jsonl(header=['file_path', 'caption'])
    jdict.export('sample_1', json_path='sample.jsonl')

    ck = ChunkDatatset('./sample_1', chunk_size=15, shuffle=True)
    print(len(ck))
    for i in range(len(ck)):
        print('='*20)
        ck.inject(i)
        for data in ck:
            print(data)

