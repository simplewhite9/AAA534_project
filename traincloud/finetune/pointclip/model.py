import torch
import torch.nn as nn
from .clip import clip
from .projection import Realistic_Projection
import numpy as np    

def load_clip(args):
    url = clip._MODELS[args.model]
    model_path = clip._download(url)

    # state_dict = torch.load('../ckpt/vitb16/pytorch_model.bin', map_location='cpu')

    try:
        # loading JIT archive
        # model = torch.jit.load(model_path, map_location='cpu').eval()
        model = torch.jit.load(model_path, map_location='cpu')
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())
    return model


class PointCLIPV2_ZS(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        model, preprocess = clip.load(args.model)
        model.eval()
        model = model.cuda()

        self.clip = model
        self.preprocess = preprocess
        self.visual_encoder = self.clip.encode_image
        self.text_encoder = self.clip.encode_text
        # self.logit_scale = self.clip.logit_scale * 0.2
        self.dtype = self.clip.dtype
        self.channel = self.clip.embed_dim

        self.num_views = 1

        # self.angle_bias = torch.tensor((np.pi / 4, np.pi / 4, np.pi / 4))
        self.pc_views = Realistic_Projection(args)
        self.finalrot = nn.Parameter(self.pc_views.finalRot, requires_grad=True)
        self.finalrot1 = nn.Parameter(self.pc_views.finalRot, requires_grad=True)
        self.get_img = self.pc_views.get_img
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)        

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.cuda()

        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        textual_encoder = Textual_Encoder(cfg, classnames, clip_model)
        
        text_feat = textual_encoder()
        self.text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.channel = cfg.MODEL.BACKBONE.CHANNEL
    
        # Realistic projection
        # self.num_views = cfg.MODEL.PROJECT.NUM_VIEWS
        self.num_views = 1
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img

        # Store features for post-search
        self.feat_store = []
        self.label_store = []
        
        self.view_weights = torch.Tensor(best_prompt_weight['{}_{}_test_weights'.format(self.cfg.DATASET.NAME.lower(), self.cfg.MODEL.BACKBONE.NAME2)]).cuda()

    # def real_proj(self, pc, imsize=224):
    #     img = self.get_img(pc).cuda() # 160, 3, 110, 110
    #     # img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)        
    #     img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)        
    #     return img
    
    def model_inference(self, pc, label=None):
        # pc : (16, 1024, 3)
        # with torch.no_grad():

        # Realistic Projection
        images = self.real_proj(pc) # 16, 1024, 3  = > 160, 3, 224, 224 (num_views = 10)
        images = images.type(self.dtype)
        image_feat = self.visual_encoder(images) # 160, 512
        
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        # image_feat_w = image_feat.reshape(-1, self.num_views, self.channel) * self.view_weights.reshape(1, -1, 1) # 16, 10, 512
        image_feat_w = image_feat.reshape(-1, self.num_views, self.channel) * self.view_weights.reshape(1, -1, 1) # 16, 10, 512
        image_feat_w = image_feat_w.reshape(-1, self.num_views * self.channel).type(self.dtype) # 16, 5120
        image_feat = image_feat.reshape(-1, self.num_views * self.channel)

        # Store for zero-shot
        self.feat_store.append(image_feat)
        self.label_store.append(label)

        logits = 100. * image_feat_w @ self.text_feat.t() # text_feats => 40, 5120
        return logits
    

    def forward(self, pc, text_id=None):
        bs = pc.shape[0]
        images1, images2 = self.get_img(pc, rotation=[self.finalrot, self.finalrot1], addrotation=self.args.addrotation)

        images1 = torch.nn.functional.interpolate(images1, size=(224, 224), mode='bilinear', align_corners=True)  
        image_feat1 = self.visual_encoder(images1.type(self.dtype)) # 160, 512
        image_feat1 = image_feat1 / image_feat1.norm(dim=1, keepdim=True)
        image_feat = image_feat1.clone()

        if images2 != None:
            images2 = torch.nn.functional.interpolate(images2, size=(224, 224), mode='bicubic', align_corners=True)  
            image_feat2 = self.visual_encoder(images2.type(self.dtype)) # 160, 512
            image_feat2 = image_feat2 / image_feat2.norm(dim=1, keepdim=True)
            image_feat = (image_feat + image_feat2)

        text_feat = self.text_encoder(text_id.squeeze(1))
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = (self.clip.logit_scale* 0.2).exp()
        logits = logit_scale * image_feat @ text_feat.t()

        pc_label = torch.arange(bs).cuda()
        loss = self.criterion(logits.reshape(-1, bs), pc_label)
        return logits, loss