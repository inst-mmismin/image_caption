import mobileclip

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

from module.projection import load_proj

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
    projection = load_proj(args.proj_type).to(device)

    return clip.to(device), llm.to(device), llm_tokenizer, projection.to(device)