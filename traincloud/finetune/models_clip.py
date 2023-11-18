import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
import pdb
from util.pos_embed import get_2d_sincos_pos_embed
import json
# from llama import ModelArgs, Tokenizer, LLaMA, Transformer
from pointclip import PointCLIPV2_ZS





def clip_model(args, **kwargs):

    print("*"*10, "Loading Model", "*"*10)
    model = PointCLIPV2_ZS(args)

    # import pdb; pdb.set_trace()
    for name, param in model.named_parameters():
        
        if 'finalrot' in name :
            param.requires_grad = True
            print(f"Train parameter {name}", "*" * 10)
        else:
            # param.requires_grad = False
            param.requires_grad = False

    return model

# set recommended archs
clip_model = clip_model
