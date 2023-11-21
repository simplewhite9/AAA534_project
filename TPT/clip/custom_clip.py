
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.datasets_3d import datasets_3d
from data.cls_to_names import *

_tokenizer = _Tokenizer()
class_prompts = ['A three-dimensional model of an airplane composed of gray, fuzzy balls.', 'A lumpy 3D model of a slanted bathtub made of dull gray balls.', 'A bed is typically represented by a rectangular shape on a grayscale map.', 'A bench can be identified from a grayscale map by its shape.', 'An unclear depth map in shades of gray of a bookshelf model at a slanted angle.', 'A bottle generally has a cylindrical shape and a narrow neck.', 'A 3D model of a bowl composed of gray balls that are difficult to see.', 'This is a depth map of a car 3D model, generated by depth sensing cameras.', 'The chair would be a dark object on the grayscale map.', 'A 3D model of a cone would look like a cone shape.', 'An obscure cup was found at the depth map.', 'There is less light in an obscure depth map of a curtain, so the features are not as clearly defined.', 'An obscure depth map of a desk would likely show a few objects on the desk in great detail, while the rest of the desk would be less detailed and more blurry.', 'This sketch depth map shows the door of a room.', 'A white heightmap in a black background of a dresser would look like a white rectangle in the center of the dresser with a black border.', 'A depth map of a flower pot can be identified by its shading.', 'A glass box will appear as a bright white region in a depth map.', 'I am looking at a 3D model of a guitar.', '3D render of a keyboard with heightmap.', 'There is an obscure depth map of a lamp.', 'The depth map of a laptop 3D model is a top-down view of the laptop, showing its various components in different colors.', 'The left or right view depth map of a white mantel would look like a white rectangle with some depth to it.', 'A grayscale image of a monitor.', 'A white, porous depth map of a night stand might look like a ghostly image of the furniture piece, with its contours and dimensions visible but slightly blurred.', 'A heightmap of a person, showing their height at different points along their body.', 'The piano would be the blackest object on the grayscale map.', 'This depth map is of a plant against a black background and is full of pores.', 'A typical radiolooks like a rectangular box with a handle on the top.', 'There is an obscure depth map of a range hood.', 'A depth map of a sink 3D model can be quite obscure, as the sink is often hidden behind other objects in a room.', 'This is a depth map of a 3D model of a sofa.', 'Thestairs3Dmodelhas a Depth Map that is quite Obscure .', 'A stool can vary in shape and size, but typically it is a small, round object that is used for sitting.', 'A 3D model of a table typically looks like a rectangular object with four legs.', 'A grayscale or white depth map of a tent 3D model would show the tent as a white object against a black background.', 'The depth map of a white toilet would appear as a white object with a black outline.', 'The image is a depth map of a 3D model of a TV stand.', 'A grayscale or white depth map of a vase 3D model would show the contours of the vase in shades of gray or white, with the darkest areas representing the deepest parts of the vase.', 'A grayscale or white depth map of a wardrobe 3D model would show the overall shape and form of the wardrobe, as well as the depth of each component.', 'A grayscale depth map of a 3D Xbox model would show a range of light to dark gray tones, with the darkest areas representing the closest parts of the model to the viewer, and the lightest areas representing the farthest parts of the model.']


DOWNLOAD_ROOT='~/.cache/clip'

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx

            n_ctx = max([len(i.split(" ")) for i in class_prompts])
            # n_ctx = len(ctx_init.split(" "))
            
            tokenized_prompts = torch.cat([tokenize(p) for p in class_prompts]).to(self.device)
            # prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            ctx_vectors = embedding[:, 1 : 1 + n_ctx, :]
            prompt_prefix = class_prompts
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        # if self.batch_size is not None: 
        #     ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            # prompts = class_prompts 
            # prompts = [class_prompts[i] for i, c in enumerate(classnames)]
            # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        # tokenized_prompts = torch.cat([tokenize(p) for p in class_prompts]).to(self.device)
        # with torch.no_grad():
            # embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames


    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            # prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
            prompts = class_prompts
            # prompts = torch.cat([tokenize(p) for p in class_prompts]).to(self.device)
        
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion

        # self.image_proj = nn.Parameter(torch.ones(1, 10))
        self.image_proj = nn.Linear(10, 1, bias=False)
        self.image_proj.weight = nn.Parameter(torch.tensor([0.75, 0.75, 0.75, 0.25, 0.75, 1.0, 0.25, 1.0, 0.75, 0.25])[None, :])
        # self.image_proj = nn.Parameter([0.75, 0.75, 0.75, 0.25, 0.75, 1.0, 0.25, 1.0, 0.75, 0.25])
        # self.image_proj.bias = nn.Parameter(torch.ones(1))

        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)
        return torch.mean(text_features, dim=0)

    def inference(self, image, inference=False):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        numImg = float(image_features.shape[0])
        text_features = self.get_text_features()

        # breakpoint()

        # image_features = (self.image_proj[None, :] @ image_features)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = self.image_proj(image_features.T).T
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)


        # if inference:
        #     image_features = (self.image_proj[None, :] @ image_features) 

        ###########################################################
        # image_features = (self.image_proj @ image_features) / numImg
        # image_features = image_features.mean(0, keepdim=True)
        #########################################################

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        # image_features = (self.image_proj @ image_features) / numImg
        # logits1 = logit_scale * image_features @ text_features.t()

        # logits = (self.image_proj @ logits ) / numImg
        return logits


    def forward(self, input, inference):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input, inference)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set in datasets_3d:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model