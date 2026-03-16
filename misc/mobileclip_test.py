import sys
sys.path.append("./ml-mobileclip")

import torch
from PIL import Image
import mobileclip

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_b',
                                                             pretrained="./checkpoints/clip/mobileclip_b.pt")
tokenizer = mobileclip.get_tokenizer('mobileclip_b')

image = preprocess(Image.open("misc/cat_example.jpg").convert('RGB')).unsqueeze(0)
text = tokenizer(["a cat", "a dog", "a bird"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    print(image_features.shape) # torch.Size([1, 512])
    print(text_features.shape) # torch.Size([3, 512])

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs) # Label probs: tensor([[9.9999e-01, 5.6537e-06, 1.3045e-07]])

'''
    CLIP(

    (image_encoder): MCi(

        (model): VisionTransformer(

        (patch_emb): Sequential(

            (0): ConvNormAct(

            (block): Sequential(

                (conv): Conv2d(3, 192, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), bias=False)

                (norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

                (act): GELU(approximate='none')

            )

            )

            (1): ConvNormAct(

            (block): Sequential(

                (conv): Conv2d(192, 192, kernel_size=(2, 2), stride=(2, 2), bias=False)

                (norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

                (act): GELU(approximate='none')

            )

            )

            (2): ConvNormAct(

            (block): Sequential(

                (conv): Conv2d(192, 768, kernel_size=(2, 2), stride=(2, 2))

            )

            )

        )

        (post_transformer_norm): LayerNormFP32((768,), eps=1e-05, elementwise_affine=True)

        (transformer): Sequential(

            (0): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (1): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (2): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (3): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (4): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (5): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (6): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (7): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (8): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (9): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (10): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

            (11): TransformerEncoder(embed_dim=768, ffn_dim=3072, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

        )

        (classifier): SimpleImageProjectionHead()

        (pos_embed): LearnablePositionalEmbedding(num_embeddings=196, embedding_dim=768, padding_idx=None)

        (emb_dropout): Dropout(p=0.0, inplace=False)

        )

    )

    (text_encoder): TextTransformer(

        (embedding_layer): Embedding(49408, 512)

        (positional_embedding): LearnablePositionalEmbedding(num_embeddings=77, embedding_dim=512, padding_idx=None)

        (embedding_dropout): Dropout(p=0.0, inplace=False)

        (transformer): ModuleList(

        (0-11): 12 x TransformerEncoder(embed_dim=512, ffn_dim=2048, dropout=0.0, ffn_dropout=0.0, stochastic_dropout=0.0, attn_fn=MultiHeadAttention, act_fn=GELU, norm_fn=layer_norm_fp32)

        )

        (final_layer_norm): LayerNormFP32((512,), eps=1e-05, elementwise_affine=True)

    )

    )
'''