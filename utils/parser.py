import argparse

def step1_train_parser():
    parser = argparse.ArgumentParser()
    
    # 학습 세팅 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

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

    return parser.parse_args()
