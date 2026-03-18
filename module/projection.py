import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    """1층 Linear: 512 -> 576"""
    def __init__(self, clip_dim=512, llm_dim=576, use_layer_norm=False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, llm_dim),
            nn.LayerNorm(llm_dim) if use_layer_norm else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPProjection(nn.Module):
    """2층 Linear + GELU: 512 -> 1024 -> 576"""
    def __init__(self, clip_dim=512, hidden_dim=1024, llm_dim=576, use_layer_norm=False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim) if use_layer_norm else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def load_proj(proj_type, clip_dim=512, llm_dim=576, use_layer_norm=False):
    if proj_type == "linear":
        projection =  LinearProjection(clip_dim=clip_dim, 
                                        llm_dim=llm_dim,
                                        use_layer_norm=use_layer_norm)
    elif proj_type == "mlp":
        projection =  MLPProjection(clip_dim=clip_dim, 
                                    hidden_dim=1024, llm_dim=llm_dim,
                                    use_layer_norm=use_layer_norm)
    
    return projection