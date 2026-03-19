import json
import os

import mobileclip
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from env import CLIP_CHECKPOINT, LLM_CHECKPOINT
from module.projection import load_proj

# step1 의 args.json 로드
def load_step1_args_from_ckpt(projection_ckpt_path):
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(projection_ckpt_path)))
    args_path = os.path.join(run_dir, "args.json")
    with open(args_path, "r", encoding="utf-8") as f:
        return json.load(f)


# step2 학습 시 step1 args에서 proj_type, use_layer_norm 추출
def _get_projection_config(projection_ckpt_path):
    step1_args = load_step1_args_from_ckpt(projection_ckpt_path)
    return step1_args["proj_type"], step1_args.get("use_layer_norm", False)


def _infer_projection_config_from_state_dict(state_dict_path):
    try:
        sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(state_dict_path, map_location="cpu")
    keys = list(sd.keys())
    has_proj_2 = any("proj.2" in k for k in keys)
    proj_type = "mlp" if has_proj_2 else "linear"
    use_layer_norm = "proj.1.weight" in keys or "proj.3.weight" in keys
    return proj_type, use_layer_norm

def load_clip(checkpoint_path, with_freeze = False):
    clip, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_b',
                                                                 pretrained=checkpoint_path)
    clip_tokenizer = mobileclip.get_tokenizer('mobileclip_b')

    if with_freeze:
        for param in clip.parameters():
            param.requires_grad = False

    return clip, clip_tokenizer, preprocess


def load_llm(checkpoint_path, with_freeze = False):
    llm_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True) 
    llm = AutoModelForCausalLM.from_pretrained(checkpoint_path)

    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    if with_freeze:
        for param in llm.parameters():
            param.requires_grad = False

    return llm, llm_tokenizer


def load_transform(args):
    if args.dataset in ("flickr8k", "flickr30k"):
        image_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_transform
    return None

def load_dataset(root, transform, args, split): # split : train OR val OR all
    if args.dataset == "flickr8k":
        from module.flickr import Flickr8k
        return Flickr8k(root, img_transform=transform, split=split)
    else: 
        raise ValueError(f"Dataset {args.dataset} not found")
    

def load_loader(root, args, split=None):
    transform = load_transform(args)
    dataset = load_dataset(root, transform, args, split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"))
    return loader

def load_step1_models(clip_checkpoint, llm_checkpoint, args, device):
    clip, _, _ = load_clip(clip_checkpoint, with_freeze=True)
    llm, llm_tokenizer = load_llm(llm_checkpoint, with_freeze=True)
    projection = load_proj(args.proj_type, use_layer_norm=args.use_layer_norm).to(device)

    return clip.to(device), llm.to(device), llm_tokenizer, projection.to(device)


def load_step2_models(clip_checkpoint, llm_checkpoint, projection_ckpt_path, args, device): 
    step1_args = load_step1_args_from_ckpt(projection_ckpt_path) 
    proj_type = step1_args["proj_type"] 
    use_layer_norm = step1_args.get("use_layer_norm", False) 

    clip, _, _ = load_clip(clip_checkpoint, with_freeze=not args.no_freeze_clip) 
    llm, llm_tokenizer = load_llm(llm_checkpoint, with_freeze=False) 
    projection = load_proj(proj_type, use_layer_norm=use_layer_norm) 
    projection.load_state_dict(torch.load(projection_ckpt_path, map_location=device)) 

    if not args.freeze_llm:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        )
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()

    return clip.to(device), llm.to(device), llm_tokenizer, projection.to(device)


# 추론용 모델 준비 
def load_inference_models(step2_run_dir, device):
    step2_run_dir = os.path.abspath(step2_run_dir)
    ckpt_dir = os.path.join(step2_run_dir, "checkpoints")
    projection_pt = os.path.join(ckpt_dir, "model_best.pt")

    with open(os.path.join(step2_run_dir, "args.json"), "r", encoding="utf-8") as f:
        step2_args = json.load(f)

    if "proj_type" in step2_args and "use_layer_norm" in step2_args:
        proj_type = step2_args["proj_type"]
        use_layer_norm = step2_args["use_layer_norm"]
    else:
        proj_type, use_layer_norm = _infer_projection_config_from_state_dict(projection_pt)

    clip, _, _ = load_clip(os.path.abspath(CLIP_CHECKPOINT), with_freeze=True)
    llm, tokenizer = load_llm(os.path.abspath(LLM_CHECKPOINT), with_freeze=True)
    projection = load_proj(proj_type, use_layer_norm=use_layer_norm)

    projection.load_state_dict(torch.load(projection_pt, map_location=device))
    lora_path = os.path.join(ckpt_dir, "lora_adapter")
    if os.path.isdir(lora_path):
        llm = PeftModel.from_pretrained(llm, lora_path)

    args_for_transform = SimpleNamespace(dataset=step2_args.get("dataset", "flickr8k"))
    transform = load_transform(args_for_transform)

    clip.eval()
    llm.eval()
    projection.eval()
    return clip.to(device), llm.to(device), projection.to(device), tokenizer, transform