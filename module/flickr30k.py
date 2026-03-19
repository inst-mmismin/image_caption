"""
Flickr30k Dataset
- Flickr8k train.txt에 있는 이미지는 제외 (학습 데이터 누수 방지)
- Flickr8k: 1000268201_693b08cb0e.jpg → numeric ID: 1000268201
- Flickr30k: 1000268201.jpg → numeric ID: 1000268201
"""
import os
from torch.utils.data import Dataset
from PIL import Image


def _get_flickr8k_train_ids(flickr8k_train_path):
    if not flickr8k_train_path or not os.path.isfile(flickr8k_train_path):
        return set()
    ids = set()
    with open(flickr8k_train_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "1000268201_693b08cb0e.jpg" -> "1000268201"
            parts = line.split("_")
            if parts:
                ids.add(parts[0])
    return ids

# Flickr30k 데이터셋 (학습용 Flickr8k 데이터 제외)
class Flickr30k(Dataset):
    def __init__(self, root, img_transform=None, txt_transform=None, 
                 flickr8k_train_path="./dataset/flickr8k/splits/train.txt"):
        self.root = root
        self.img_transform = img_transform
        self.txt_transform = txt_transform

        # Flickr8k train 제외할 ID 목록
        self.exclude_ids = _get_flickr8k_train_ids(flickr8k_train_path)
        
        # captions.txt 로드 
        self.captions = {}
        captions_path = os.path.join(self.root, "captions.txt")
        with open(captions_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                idx = line.find("jpg,")
                if idx == -1:
                    continue
                image_id = line[: idx + 3].strip()
                caption = line[idx + 4 :].strip().strip('"').strip()
                if image_id not in self.captions:
                    self.captions[image_id] = []
                self.captions[image_id].append(caption)

        # Images 폴더에서 이미지 목록 
        images_dir = os.path.join(self.root, "Images")
        all_images = sorted(os.listdir(images_dir))
        self.images = []
        for img in all_images:
            if img not in self.captions:
                continue
            if self.exclude_ids:
                numeric_id = img.replace(".jpg", "").replace(".JPG", "")
                if numeric_id in self.exclude_ids:
                    continue
            self.images.append(img)

        assert all(img in self.captions for img in self.images), "일부 이미지에 캡션이 없습니다"

    def get_refs(self):
        # 평가용 참조 캡션. {image_id: [cap1, cap2, ...]} 
        return {img: self.captions[img] for img in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "Images", self.images[idx])
        image = Image.open(image_path).convert("RGB")
        if self.img_transform is not None:
            image = self.img_transform(image)

        caption_list = self.captions[self.images[idx]]
        caption = caption_list[0] if caption_list else ""
        if self.txt_transform is not None:
            caption = self.txt_transform(caption)
        return image, caption
