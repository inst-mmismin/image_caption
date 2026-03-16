"""Loss functions: Contrastive Loss, LM Loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# CLIP 스타일의 Contrastive loss
# 이미지와 텍스트 사이의 유사도를 계산 
def contrastive_loss(image_features, text_features, temperature):
    image_features = image_features.float()
    text_features = text_features.float()
    
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits = (image_features @ text_features.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    # Symmetric loss(image-to-text + text-to-image) 고려 
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


# Causal LM loss
# LM 입력 : [image_prefix] + [caption] 
# 이미지 조건하에 생성되는 caption의 loss 계산 
def lm_loss(llm, input_embeds, caption_ids, attention_mask, llm_embed_layer, device):
    batch_size = input_embeds.size(0)
    num_prefix = input_embeds.size(1)

    # 텍스트 임베딩
    caption_embeds = llm_embed_layer(caption_ids)  # (B, seq_len, D)

    input_embeds = input_embeds.to(caption_embeds.dtype) # float32 -> bfloat16

    # 전체 임베딩 생성 (Concatenate: [prefix | caption])
    inputs_embeds = torch.cat([input_embeds, caption_embeds], dim=1)  # (B, num_prefix + seq_len, D)

    # 이미지 prefix를 과려한 attention_mask 생성 
    prefix_mask = torch.ones(batch_size, num_prefix, device=device, dtype=attention_mask.dtype)
    full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    # label 생성 (prefix & padding: -100, caption: actual token ids)
    prefix_labels = torch.full(
        (batch_size, num_prefix), -100, device=device, dtype=torch.long
    )
    caption_labels = torch.where(
        attention_mask.bool(), caption_ids, torch.full_like(caption_ids, -100)
    )
    labels = torch.cat([prefix_labels, caption_labels], dim=1)

    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attention_mask,
        labels=labels,
    )
    return outputs.loss
