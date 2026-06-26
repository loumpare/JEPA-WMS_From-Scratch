import torch




class patch_embedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, d_model: int , in_channels: int):
        super().__init__()