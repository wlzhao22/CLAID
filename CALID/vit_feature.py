import vision_transformer as vits
import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

def get_dino_output(img_var, device):
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    model.eval()
    model.to(device)
    output, attens,q, k, v = model.get_last_output_and_selfattention(img_var)

    return output, attens, q, k, v

def get_dinov2_output(img_var, device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()
    model.to(device)
    output = model.get_intermediate_layers(img_var, n=1)

    return output


