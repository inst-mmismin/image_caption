import os
import sys

sys.path.append("./ml-mobileclip")

import torch
from tqdm import tqdm

from env import CLIP_CHECKPOINT, LLM_CHECKPOINT
from utils.load import load_loader, load_step1_models, load_transform
from utils.parser import step1_train_parser
from utils.step1_tools import train_step
from utils.evaluate import run_eval
from utils.tensorboard import TensorBoardLogger


def main():
    args = step1_train_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logger = TensorBoardLogger(step="step1", log_dir_base=args.log_dir)
    logger.save_args(args)
    ckpt_dir = logger.get_checkpoint_dir()
    
    proj_path = os.path.join(ckpt_dir, "projection_last.pt")
    proj_best_path = os.path.join(ckpt_dir, "projection_best.pt")

    # 데이터 로드
    train_loader = load_loader(args.root, args, split="train")

    # 모델 로드
    clip_checkpoint = os.path.abspath(CLIP_CHECKPOINT)
    llm_checkpoint = os.path.abspath(LLM_CHECKPOINT)
    clip, llm, llm_tokenizer, projection = load_step1_models(
        clip_checkpoint, llm_checkpoint, args, device
    )

    transform = load_transform(args)

    # Optimizer
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.learning_rate)

    best_cider = -1.0
    global_step = 0

    # Train loop
    projection.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            images, captions_raw = batch
            images = images.to(device)

            # Feed Forward & Loss 
            loss, loss_lm_val, loss_contrastive_val = train_step(
                clip, llm, projection,
                images, captions_raw,
                llm_tokenizer, device,
                args,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # logging 
            logger.add_scalar("train/loss_step", loss.item(), global_step)
            if loss_lm_val is not None:
                logger.add_scalar("train/loss_lm", loss_lm_val.item(), global_step)
            if loss_contrastive_val is not None:
                logger.add_scalar("train/loss_contrastive", loss_contrastive_val.item(), global_step)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        avg_loss = epoch_loss / (batch_idx + 1)
        logger.add_scalar("train/loss_epoch", avg_loss, epoch)
        print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_loss:.4f}")

        # 평가 및 저장 
        cider = run_eval(clip, llm, projection, llm_tokenizer, device, args.root, transform)
        if cider is not None:
            logger.add_scalar("eval/cider", cider, epoch)
            print(f"Epoch {epoch + 1}/{args.epochs} | CIDEr: {cider:.4f}")

            if cider > best_cider:
                best_cider = cider
                torch.save(projection.state_dict(), proj_best_path)
                print(f"  -> Best CIDEr! Saved to {proj_best_path}")

        projection.train()

    # 마지막 모델 저장
    torch.save(projection.state_dict(), proj_path)
    print(f"Projection 저장: {proj_path}")
    
    if best_cider > 0:
        print(f"Best CIDEr: {best_cider:.4f} -> {proj_best_path}")

    logger.close()
    print("Training done.")


if __name__ == "__main__":
    main()
