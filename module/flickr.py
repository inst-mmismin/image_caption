import os
import random
from torch.utils.data import Dataset
from PIL import Image

class Flickr8k(Dataset):
    def __init__(self, root, img_transform=None, txt_transform=None, split=None):
        self.root = root
        all_images = sorted(os.listdir(os.path.join(self.root, "Images")))
        if split in ['train', 'val']:
            split_file = os.path.join(self.root, "splits", f"{split}.txt")
            with open(split_file, "r") as f:
                allowed = set(line.strip() for line in f if line.strip())
            self.images = [img for img in all_images if img in allowed] # split_file에 있는 이미지만
        else:
            self.images = all_images # 전체 이미지
        self.captions = {}
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        
        with open(os.path.join(self.root, "captions.txt"), "r") as f:
            lines = f.readlines()[1:] # 첫 줄은 제외
            for line in lines:
                image_id, caption = line.split("jpg,")
                image_id = image_id + 'jpg'
                if image_id not in self.captions:
                    self.captions[image_id] = []
                self.captions[image_id].append(caption)
        
        assert all(img in self.captions for img in self.images), "일부 이미지에 캡션이 없습니다"

    def get_refs(self):
        # 평가용 참조 캡션. {image_id: [cap1, cap2, ...]} 
        return {img: self.captions[img] for img in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "Images", self.images[idx])
        image = Image.open(image_path)
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        
        caption_list = self.captions[self.images[idx]]
        valid_captions = [c for c in caption_list if len(c.strip().split()) > 2]
        caption = random.choice(valid_captions if valid_captions else caption_list)
        if self.txt_transform is not None:
            caption = self.txt_transform(caption)
        return image, caption