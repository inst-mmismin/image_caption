"""Step2 학습용 유틸: caption generation LM loss"""
import torch
from module.loss import lm_loss
from env import CAPTION_PROMPT


def train_step(clip, llm, projection, images, captions_raw, llm_tokenizer, device, args):
    """한 스텝 학습: [image_prefix | prompt | caption] → LM loss"""
    caption_enc = llm_tokenizer(
        captions_raw,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    caption_ids = caption_enc["input_ids"].to(device)
    attention_mask = caption_enc["attention_mask"].to(device)

    with torch.no_grad():
        clip_img_features = clip.encode_image(images, normalize=False)

    projected_img = projection(clip_img_features)
    batch_size = projected_img.size(0)
    prefix_embeds = projected_img.unsqueeze(1)

    prompt_ids = llm_tokenizer(
        CAPTION_PROMPT,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    prompt_embeds = llm.get_input_embeddings()(prompt_ids).expand(batch_size, -1, -1)

    loss = lm_loss(
        llm=llm,
        input_embeds=prefix_embeds,
        caption_ids=caption_ids,
        attention_mask=attention_mask,
        llm_embed_layer=llm.get_input_embeddings(),
        device=device,
        prompt_embeds=prompt_embeds,
    )
    return loss
