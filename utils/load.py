import json
import os

import mobileclip
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from module.projection import load_proj

# step1 의 args.json 로드
def load_step1_args_from_ckpt(projection_ckpt_path):
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(projection_ckpt_path)))
    args_path = os.path.join(run_dir, "args.json")
    with open(args_path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    if args.dataset == "flickr8k":
        image_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_transform
    else:
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