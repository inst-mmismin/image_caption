import torch 
from module.loss import contrastive_loss, lm_loss

# Caption을 입력해서 전체 텍스트의 맥력을 담고 있는 마지막 Hidden state를 반환 
# 해당 정보가 이미지의 feature와 비교되어 Contrastive loss 계산에 사용됨 
def get_text_features(llm, caption_ids, attention_mask, device):
    
    with torch.no_grad():
        outputs = llm(
            input_ids=caption_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    
    hidden_states = outputs.hidden_states[-1]  # (B, seq_len, D)
    last_idx = attention_mask.sum(dim=1) - 1  # (B,)
    
    batch_idx = torch.arange(hidden_states.size(0), device=device)
    text_features = hidden_states[batch_idx, last_idx, :]
    return text_features


# 한 스텝 학습 처리 함수 
def train_step( 
    clip, llm, projection,
    images, captions_raw,
    llm_tokenizer, device,
    args,
):
    # Caption tokenize
    caption_enc = llm_tokenizer(
        captions_raw,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    caption_ids = caption_enc["input_ids"].to(device)
    attention_mask = caption_enc["attention_mask"].to(device)

    with torch.no_grad():
        clip_img_features = clip.encode_image(images, normalize=False)  # (B, 512)

    projected_img = projection(clip_img_features)  # (B, 576)

    total_loss = torch.tensor(0.0, device=device)
    loss_lm = None
    loss_contrastive = None

    # Contrastive loss
    if args.use_contrastive:
        text_features = get_text_features(llm, caption_ids, attention_mask, device)
        loss_contrastive = contrastive_loss(
            projected_img, text_features, temperature=args.contra_temp
        )
        total_loss = total_loss + args.weight_contrastive * loss_contrastive

    # LM loss
    if args.use_lm:
        prefix_embeds = projected_img.unsqueeze(1)
        loss_lm = lm_loss(
            llm=llm,
            input_embeds=prefix_embeds,
            caption_ids=caption_ids,
            attention_mask=attention_mask,
            llm_embed_layer=llm.get_input_embeddings(),
            device=device,
        )
        
        total_loss = total_loss + args.weight_lm * loss_lm

    return total_loss, loss_lm, loss_contrastive