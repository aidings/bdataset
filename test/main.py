import torch
import json
from bdataset import FastImageDataset, FastLineReader


if __name__ == '__main__':
    # reader = FastLineReader('sample.jsonl', parse=json.loads, skip_head=True) 
    reader = FastLineReader.from_index('sample.jsonl.index', parse=json.loads)
    for data in reader:
        print(data)

    """
    dataset = FastImageDataset([reader], 'img_path')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    """

