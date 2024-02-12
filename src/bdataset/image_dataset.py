from PIL import Image
from loguru import logger
import random
from .fast_line_dataset import FastLineDataset, InjectDataset



class FastImageDataset(FastLineDataset):
    def __init__(self, readers, img_key):
        super().__init__(readers)
        self.img_key = img_key

    def imread(self, idx):
        n = self.__len__()
        while True:
            img_dir = self.read_line(idx)[self.img_key]
            try:
                image = Image.open(img_dir).convert('RGB')
                break
            except:
                logger.warning(f'error image: {idx}-{img_dir}')
                idx = random.randint(0, n-1)
        return image
    
    def transforms(self, idx):
        return self.imread(idx)

class InjectImageDataset(InjectDataset):
    def __init__(self, img_key):
        super().__init__()
        self.img_key = img_key
    
    def imread(self, idx):
        n = self.__len__()
        while True:
            img_dir = self.datas[idx][self.img_key]
            try:
                image = Image.open(img_dir).convert('RGB')
                break
            except:
                logger.warning(f'error image: {idx}-{img_dir}')
                idx = random.randint(0, n-1)
        return image
    
    def transforms(self, idx):
        return self.imread(idx)
