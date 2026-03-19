import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pycocoevalcap.cider.cider import Cider

from env import CAPTION_PROMPT
from module.flickr import Flickr8k

from module.EvalDataset import ImageIdEvalDataset
from utils.step1_tools import get_text_features

# Instruct 모델이 생성한 질문 등 제거, 캡션만 추출
def clean_caption(raw: str) -> str:
    raw = raw.strip()
    parts = raw.replace("\r", "\n").split("\n")
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.startswith(("What ", "Who ", "Where ", "When ", "Why ", "How ")):
            continue
        return p
    return raw.split("\n")[0].strip() if raw else ""


def compute_cider(refs, hyps):
    # refs: {image_id: [ref_caption1, ref_caption2, ...]}
    # hyps: {image_id: generated_caption}
    res = {img_id: [cap] for img_id, cap in hyps.items()}
    scorer = Cider()
    score, _ = scorer.compute_score(refs, res)
    return float(score) # 0 ~ 10


def run_cider_eval(clip, llm, projection, tokenizer, device, data_root, transform, max_new_tokens=64):
    split_file = os.path.join(data_root, "splits", "val.txt")

    clip.eval()
    llm.eval()
    projection.eval()

    dataset = Flickr8k(data_root, img_transform=transform, split="val")
    loader = DataLoader(ImageIdEvalDataset(dataset), batch_size=32, shuffle=False)
    refs = dataset.get_refs()

    hyps = {}
    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)
            clip_feats = clip.encode_image(images, normalize=False)
            img_prefix = projection(clip_feats).unsqueeze(1)
            prompt_ids = tokenizer(
                CAPTION_PROMPT,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(device)
            prompt_embeds = llm.get_input_embeddings()(prompt_ids).expand(images.size(0), -1, -1)
            inputs_embeds = torch.cat([img_prefix, prompt_embeds], dim=1)
            inputs_embeds = inputs_embeds.to(prompt_embeds.dtype)
            prefix_mask = torch.ones(inputs_embeds.size(0), inputs_embeds.size(1), device=device, dtype=torch.long)
            outputs = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prefix_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            gen_ids = outputs[:, inputs_embeds.size(1):]
            for i, ids in enumerate(gen_ids):
                raw = tokenizer.decode(ids, skip_special_tokens=True).strip()
                hyps[image_ids[i]] = clean_caption(raw)

    common = set(refs.keys()) & set(hyps.keys())
    refs = {k: refs[k] for k in common}
    hyps = {k: hyps[k] for k in common}
    if not refs:
        return None

    return compute_cider(refs, hyps)

# positive pair의 cos similarity, in-batch 상황의 정답이 1위인 비율 & top5 중 정답이 있는 비율 계산 
def run_retrieval_eval(clip, llm, projection, tokenizer, device, data_root, transform):
    split_file = os.path.join(data_root, "splits", "val.txt")
    if not os.path.isfile(split_file):
        return None

    clip.eval()
    llm.eval()
    projection.eval()

    dataset = Flickr8k(data_root, img_transform=transform, split="val")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    sims, recall1_list, recall5_list = [], [], []

    with torch.no_grad():
        for images, captions_raw in loader:
            images = images.to(device)
            enc = tokenizer(
                list(captions_raw) if isinstance(captions_raw, (list, tuple)) else captions_raw,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            caption_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            img_feats = projection(clip.encode_image(images, normalize=False))
            text_feats = get_text_features(llm, caption_ids, attention_mask, device)

            img_feats = F.normalize(img_feats.float(), dim=-1)
            text_feats = F.normalize(text_feats.float(), dim=-1)
            sim = img_feats @ text_feats.T  # (B, B)

            # Positive pair similarity (diagonal)
            sims.append(torch.diag(sim).mean().item())

            # Recall@1: argmax == correct index
            pred = sim.argmax(dim=1)
            recall1_list.append((pred == torch.arange(sim.size(0), device=device)).float().mean().item())

            # Recall@5: correct in top 5
            _, top5 = sim.topk(5, dim=1)
            correct = torch.arange(sim.size(0), device=device).unsqueeze(1).expand_as(top5)
            recall5_list.append((top5 == correct).any(dim=1).float().mean().item())

    return {
        "mean_sim": sum(sims) / len(sims) if sims else 0.0,
        "recall1": sum(recall1_list) / len(recall1_list) if recall1_list else 0.0,
        "recall5": sum(recall5_list) / len(recall5_list) if recall5_list else 0.0,
    }
