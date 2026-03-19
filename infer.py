import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting.*")

sys.path.append("./ml-mobileclip")

import torch
from PIL import Image

from env import CAPTION_PROMPT
from utils.load import load_inference_models
from utils.evaluate import clean_caption


def parse_args():
    parser = argparse.ArgumentParser(description="이미지 캡션 추론")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Step2 run 폴더")
    parser.add_argument("--image", type=str, required=True,
                        help="입력 이미지 경로")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="생성 최대 토큰 수")
    return parser.parse_args()


def generate_caption(clip, llm, projection, tokenizer, image, transform, device, max_new_tokens=64):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        clip_feats = clip.encode_image(image_tensor, normalize=False)
        img_prefix = projection(clip_feats).unsqueeze(1)
        prompt_ids = tokenizer(
            CAPTION_PROMPT,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        prompt_embeds = llm.get_input_embeddings()(prompt_ids).expand(1, -1, -1)
        inputs_embeds = torch.cat([img_prefix, prompt_embeds], dim=1)
        inputs_embeds = inputs_embeds.to(prompt_embeds.dtype)
        prefix_mask = torch.ones(1, inputs_embeds.size(1), device=device, dtype=torch.long)

        outputs = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prefix_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    gen_ids = outputs[:, inputs_embeds.size(1):]
    raw = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    return clean_caption(raw)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip, llm, projection, tokenizer, transform = load_inference_models(args.ckpt_dir, device)

    if not os.path.isfile(args.image):
        print(f"이미지를 찾을 수 없습니다: {args.image}")
        return

    caption = generate_caption(
        clip, llm, projection, tokenizer,
        args.image, transform, device,
        max_new_tokens=args.max_new_tokens,
    )
    print(caption)


if __name__ == "__main__":
    main()
