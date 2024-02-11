from .chunk_dataset import ChunkDatatset
from PIL import Image
from loguru import logger
import random

class ImageDataset(ChunkDatatset):
    def __init__(self, root_path, image_index=0, chunk_size=None, shuffle=False):
        super().__init__(root_path, chunk_size, shuffle)
        self.image_index = image_index

    def imread(self, idx):
        n = len(self.datas[self.image_index])
        while True:
            img_dir = self.datas[self.image_index][idx]
            try:
                image = Image.open(img_dir).convert('RGB')
                break
            except:
                logger.warning(f'error image: {img_dir}')
                idx = random.randint(0, n-1)
        return image
    
    def __getitem__(self, idx):
        return self.imread(idx)
        
        

