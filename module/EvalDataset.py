import torch


class ImageIdEvalDataset(torch.utils.data.Dataset):
    # (image, caption) 반환하는 데이터셋을 (image, image_id) 반환하도록 변환.
    # 데이터셋에 .images 속성이 필요.
    
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return img, self.base.images[idx]
