import argparse

def step1_train_parser():
    parser = argparse.ArgumentParser()
    
    # 학습 세팅 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Cosine annealing 최소 learning rate (eta_min)")

    # 로깅 세팅
    parser.add_argument("--log_dir", type=str, default="./runs",
                        help="TensorBoard log 저장 경로")
    # 데이터 세팅 
    parser.add_argument("--dataset", type=str, default="flickr8k")
    parser.add_argument("--root", type=str, default="./dataset/flickr8k")
    parser.add_argument("--splits_dir", type=str, default="./dataset/flickr8k/splits",
                        help="train.txt, val.txt, all.txt가 있는 폴더")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"],
                        help="사용할 split (train/val/all)")

    # 모델 세팅
    parser.add_argument("--proj_type", type=str, default="mlp",
                        choices=["linear", "mlp"],
                        help="사용할 Projection 종류 - linear (1층) or mlp (2층+GELU)")
    parser.add_argument("--use_contrastive", action="store_true", default=False,
                        help="Contrastive loss 사용 여부")
    parser.add_argument("--weight_contrastive", type=float, default=0.1,
                        help="Contrastive loss 가중치")
    parser.add_argument("--use_lm", action="store_true", default=False,
                        help="LM loss 사용 여부")
    parser.add_argument("--weight_lm", type=float, default=1.0,
                        help="LM loss 가중치")
    parser.add_argument("--contra_temp", type=float, default=0.07,
                        help="logit 분포 스케일링 파라미터")
    parser.add_argument("--use_layer_norm", action="store_true", default=False,
                        help="Projection 내 LayerNorm 사용 여부")

    return parser.parse_args()


def step2_train_parser():
    parser = argparse.ArgumentParser()

    # 학습 세팅
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--projection_lr", type=float, default=1e-4,
                        help="Projection learning rate")
    parser.add_argument("--lora_lr", type=float, default=5e-5,
                        help="LoRA learning rate (freeze_llm=False일 때)")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Cosine annealing 최소 learning rate (eta_min)")

    # 로깅 세팅
    parser.add_argument("--log_dir", type=str, default="./runs",
                        help="TensorBoard log 저장 경로")

    # 데이터 세팅
    parser.add_argument("--dataset", type=str, default="flickr8k")
    parser.add_argument("--root", type=str, default="./dataset/flickr8k")

    # 모델 세팅
    parser.add_argument("--projection_ckpt", type=str, required=True,
                        help="Step1 projection checkpoint 경로")
    parser.add_argument("--no_freeze_clip", action="store_true",
                        help="CLIP 학습 (기본: freeze)")
    parser.add_argument("--freeze_llm", action="store_true",
                        help="LLM freeze (기본: 학습)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Caption 최대 토큰 길이")

    # PEFT (LoRA) 
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (scaling)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj",
                        help="LoRA target modules (쉼표 구분, 모델에 맞게 조정)")

    # Accelerate
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision (bf16 권장)")

    return parser.parse_args()
