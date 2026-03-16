import os
import torch
from torch.utils.data import DataLoader

from pycocoevalcap.cider.cider import Cider

from module.flickr import Flickr8k


def compute_cider(refs, hyps):
    # refs: {image_id: [ref_caption1, ref_caption2, ...]}
    # hyps: {image_id: generated_caption}
    res = {img_id: [cap] for img_id, cap in hyps.items()}
    scorer = Cider()
    score, _ = scorer.compute_score(refs, res)
    return float(score) # 0 ~ 10


def load_refs(data_root, split_file):
    refs = {}
    with open(os.path.join(data_root, "captions.txt"), "r") as f:
        for line in f.readlines()[1:]: # 첫 줄은 제외
            parts = line.strip().split("jpg,", 1)
            if len(parts) != 2:
                continue
            image_id = parts[0] + "jpg"
            caption = parts[1].strip()
            if image_id not in refs:
                refs[image_id] = []
            refs[image_id].append(caption)
    if split_file and os.path.isfile(split_file):
        with open(split_file, "r") as f:
            allowed = set(line.strip() for line in f if line.strip())
        refs = {k: v for k, v in refs.items() if k in allowed}
    return refs


def run_eval(clip, llm, projection, tokenizer, device, data_root, transform, max_new_tokens=64):
    split_file = os.path.join(data_root, "splits", "val.txt")

    clip.eval()
    llm.eval()
    projection.eval()

    dataset = Flickr8k(data_root, img_transform=transform, split="val")

    class EvalDataset(torch.utils.data.Dataset):
        def __init__(self, flickr):
            self.flickr = flickr
        def __len__(self):
            return len(self.flickr)
        def __getitem__(self, idx):
            img, _ = self.flickr[idx]
            return img, self.flickr.images[idx] # image, image_id (key값으로 사용)

    loader = DataLoader(EvalDataset(dataset), batch_size=32, shuffle=False)
    refs = load_refs(data_root, split_file)

    hyps = {}
    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)
            clip_feats = clip.encode_image(images, normalize=False)
            prefix = projection(clip_feats).unsqueeze(1)
            outputs = llm.generate(
                inputs_embeds=prefix,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            gen_ids = outputs[:, 1:]
            for i, ids in enumerate(gen_ids):
                cap = tokenizer.decode(ids, skip_special_tokens=True).strip()
                hyps[image_ids[i]] = cap

    common = set(refs.keys()) & set(hyps.keys())
    refs = {k: refs[k] for k in common}
    hyps = {k: hyps[k] for k in common}
    if not refs:
        return None

    return compute_cider(refs, hyps)
