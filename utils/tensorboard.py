import json
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, step, log_dir_base = "./runs"):
        self.step = step
        self.log_dir = os.path.join(log_dir_base, 
                                    step, 
                                    datetime.now().strftime("%Y%m%d_%H%M%S")) 
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard log: {self.log_dir}")

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag, tag_scalar_dict, step):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def save_args(self, args):
        path = os.path.join(self.log_dir, "args.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, 
                      indent=4, ensure_ascii=False, default=str)

    def get_checkpoint_dir(self):
        ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    def close(self):
        self.writer.close()
