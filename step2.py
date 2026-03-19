import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting.*")

sys.path.append("./ml-mobileclip")

import torch
from tqdm import tqdm
from accelerate import Accelerator

from env import CLIP_CHECKPOINT, LLM_CHECKPOINT
from utils.load import load_loader, load_step2_models, load_transform, _get_projection_config
from utils.parser import step2_train_parser
from utils.step2_tools import train_step
from utils.evaluate import run_cider_eval
from utils.tensorboard import TensorBoardLogger


def main():
    args = step2_train_parser()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"Using device: {device}")
        print(f"Mixed precision: {args.mixed_precision} | Accumulation steps: {args.gradient_accumulation_steps}")

    logger = TensorBoardLogger(step="step2", log_dir_base=args.log_dir)
    if accelerator.is_main_process:
        args.proj_type, args.use_layer_norm = _get_projection_config(args.projection_ckpt)
        logger.save_args(args)
    ckpt_dir = logger.get_checkpoint_dir()

    model_last_path = os.path.join(ckpt_dir, "model_last.pt")
    model_best_path = os.path.join(ckpt_dir, "model_best.pt")
    lora_adapter_dir = os.path.join(ckpt_dir, "lora_adapter")

    # 데이터 로드
    train_loader = load_loader(args.root, args, split="train")
    transform = load_transform(args)

    # 모델 로드
    clip_checkpoint = os.path.abspath(CLIP_CHECKPOINT)
    llm_checkpoint = os.path.abspath(LLM_CHECKPOINT)
    projection_ckpt = os.path.abspath(args.projection_ckpt)
    clip, llm, llm_tokenizer, projection = load_step2_models(
        clip_checkpoint, llm_checkpoint, projection_ckpt, args, device
    )

    # Optimizer: projection + (LLM LoRA if not frozen), 각각 다른 lr
    param_groups = [{"params": projection.parameters(), "lr": args.projection_lr}]
    if not args.freeze_llm:
        param_groups.append({"params": llm.parameters(), "lr": args.lora_lr})
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    clip, llm, projection, optimizer, scheduler, train_loader = accelerator.prepare(
        clip, llm, projection, optimizer, scheduler, train_loader
    )

    best_cider = -1.0
    global_step = 0

    # Train loop
    projection.train()
    if not args.freeze_llm:
        llm.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", disable=not accelerator.is_main_process)
        for batch_idx, batch in enumerate(pbar):
            images, captions_raw = batch

            loss = train_step(
                clip, llm, projection,
                images, captions_raw,
                llm_tokenizer, device,
                args,
            )
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.detach().item()
            if accelerator.is_main_process:
                logger.add_scalar("train/loss_step", loss.item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / (batch_idx + 1)
        if accelerator.is_main_process:
            logger.add_scalar("train/loss_epoch", avg_loss, epoch)
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_loss:.4f}")

        # 평가 
        if accelerator.is_main_process:
            unwrap_clip = accelerator.unwrap_model(clip)
            unwrap_llm = accelerator.unwrap_model(llm)
            unwrap_proj = accelerator.unwrap_model(projection)
            cider = run_cider_eval(unwrap_clip, unwrap_llm, unwrap_proj, llm_tokenizer, device, args.root, transform)
            if cider is not None:
                logger.add_scalar("eval/cider", cider, epoch)
                print(f"Epoch {epoch + 1}/{args.epochs} | CIDEr: {cider:.4f}")

                if cider > best_cider:
                    best_cider = cider
                    torch.save(unwrap_proj.state_dict(), model_best_path)
                    if not args.freeze_llm:
                        unwrap_llm.save_pretrained(lora_adapter_dir)
                    print(f"  -> Best CIDEr! Saved to {model_best_path}")

        scheduler.step()
        if accelerator.is_main_process:
            logger.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        projection.train()
        if not args.freeze_llm:
            llm.train()

    # 마지막 모델 저장
    if accelerator.is_main_process:
        unwrap_proj = accelerator.unwrap_model(projection)
        torch.save(unwrap_proj.state_dict(), model_last_path)
        if not args.freeze_llm:
            unwrap_llm = accelerator.unwrap_model(llm)
            unwrap_llm.save_pretrained(os.path.join(ckpt_dir, "lora_adapter_last"))
        print(f"Model 저장: {model_last_path}")
        print(f"Best CIDEr: {best_cider:.4f} -> {model_best_path}")

    logger.close()
    if accelerator.is_main_process:
        print("Training done.")


if __name__ == "__main__":
    main()
