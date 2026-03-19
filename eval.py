import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting.*")

sys.path.append("./ml-mobileclip")

import torch
from torch.utils.data import DataLoader

from env import CAPTION_PROMPT
from utils.load import load_inference_models
from utils.evaluate import compute_cider, clean_caption
from module.EvalDataset import ImageIdEvalDataset
from module.flickr30k import Flickr30k


def parse_args():
    parser = argparse.ArgumentParser(description="Flickr30k CIDEr 평가 (Flickr8k train 제외)")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Step2 run 폴더 (runs/step2/YYYYMMDD_HHMMSS)")
    parser.add_argument("--flickr30k_root", type=str, default="./dataset/flickr30k",
                        help="Flickr30k 데이터 경로")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="배치 크기")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="캡션 생성 최대 토큰 수")
    return parser.parse_args()


# 모델로 캡션 생성. {image_id: generated_caption}
def generate_captions(clip, llm, projection, tokenizer, loader, device, max_new_tokens=64):
    clip.eval()
    llm.eval()
    projection.eval()

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

            gen_ids = outputs[:, inputs_embeds.size(1) :]
            for i, ids in enumerate(gen_ids):
                raw = tokenizer.decode(ids, skip_special_tokens=True).strip()
                hyps[image_ids[i]] = clean_caption(raw)

    return hyps


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.flickr30k_root):
        print(f"Flickr30k 경로가 없습니다: {args.flickr30k_root}")
        return

    # 모델 로드
    print("모델 로딩...")
    clip, llm, projection, tokenizer, transform = load_inference_models(args.ckpt_dir, device)

    # Flickr30k 데이터셋 (Flickr8k train 제외)
    print("Flickr30k 데이터 로딩...")
    dataset = Flickr30k(
        root=args.flickr30k_root,
        img_transform=transform,
    )
    print(f"  평가 이미지 수: {len(dataset)}")

    loader = DataLoader(
        ImageIdEvalDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
    )
    refs = dataset.get_refs()

    # 캡션 생성
    print("캡션 생성 중...")
    hyps = generate_captions(
        clip, llm, projection, tokenizer,
        loader, device,
        max_new_tokens=args.max_new_tokens,
    )

    common = set(refs.keys()) & set(hyps.keys())
    refs = {k: refs[k] for k in common}
    hyps = {k: hyps[k] for k in common}

    if not refs:
        print("평가할 샘플이 없습니다.")
        return

    cider = compute_cider(refs, hyps)
    print(f"\n[Flickr30k 평가] (Flickr8k train 제외, N={len(refs)})")
    print(f"CIDEr: {cider:.4f}")


if __name__ == "__main__":
    main()
