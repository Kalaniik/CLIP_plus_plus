import torch
import torch.nn as nn
import torch.nn.functional as F
from clip_plus_plus.utils.clip_part import *
from clip_plus_plus.models.base import *

from clip_plus_plus.utils import build_clip
from clip_plus_plus.utils.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()

class ClipPlusPlus(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        if cfg.model_name.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model)

        self.prompt_learner = PromptLearner(cfg, cfg.classname, clip_model)
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale

        self.dtype = clip_model.dtype
        self.n_cls = len(cfg.classname)

        self.device = torch.device("cuda:{}".format(int(cfg.gpu_id)))
        
        self.vp = cfg.vp
        self.batch_size = cfg.batch_size
        if self.vp:
            self.create_visual_prompt()

    def forward(self, image, use_vp=True, prompt_weight=0.1):
        prompts = self.prompt_learner()

        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        should_use_vp = self.vp if use_vp else use_vp
        
        if should_use_vp:
            image = self.add_visual_prompt(image, prompt_weight)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    
    def create_visual_prompt(self, image=None):
        if image is not None:
            B, C, H, W = image.shape
            self.visual_prompt = nn.Parameter(
                torch.randn(B, C, H, W, device=self.device) * 0.02
            )
            print(f"Create Visual learnable prompt, shape: [{B}, {C}, {H}, {W}]")
        else:
            self.visual_prompt = nn.Parameter(
                torch.randn(self.batch_size, 3, 224, 224, device=self.device) * 0.02
            )
            print(f"Create Visual learnable prompt, shape: [{self.batch_size}, 3, 224, 224]")

    def reset_visual_prompt(self, image=None):
        if image is not None:
            B, C, H, W = image.shape
            self.visual_prompt = nn.Parameter(
                torch.randn(B, C, H, W, device=self.device) * 0.02
            )
        else:
            self.visual_prompt = nn.Parameter(
                torch.randn(self.batch_size, 3, 224, 224, device=self.device) * 0.02
            )

    def get_visual_prompt_params(self):
        """
        Trainable parameters of visual prompts
        Returns:
            list: The list containing the self.visual_prompt parameter
        """
        if hasattr(self, 'visual_prompt'):
            return [self.visual_prompt]
        else:
            return []

    def get_trainable_params(self, update_tp=True, update_vp=True):
        """
        Obtain all trainable parameters, including visual_prompt and prompt_learner parameters
        Args:
            update_tp: Whether to update the text prompt
            update_vp: Whether to update the visual prompt
        Returns:
            list: The list containing all trainable parameters
        """
        params = []
        
        # Add visual_prompt parameters
        if hasattr(self, 'visual_prompt') and update_vp:
            params.append(self.visual_prompt)
        
        # Add prompt_learner parameters
        if hasattr(self.prompt_learner, 'ctx') and self.prompt_learner.ctx is not None and update_tp:
            params.append(self.prompt_learner.ctx)
        
        return params

    def add_visual_prompt(self, image, prompt_weight):
        """
        Add Visual learnable prompt to batch
        Args:
            image: [B, 3, H, W] input image
            prompt_weight: weight for the visual prompt
        Returns:
            modified_image: [B, 3, H, W] modified image
        """

        B, C, H, W = image.shape

        # Ensure visual_prompt is on the correct device
        if self.visual_prompt.device != image.device:
            self.visual_prompt = self.visual_prompt.to(image.device)

        if self.visual_prompt.shape[2] != H or self.visual_prompt.shape[3] != W:
            visual_prompt_resized = torch.nn.functional.interpolate(
                self.visual_prompt, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            visual_prompt_resized = self.visual_prompt

        image = image + visual_prompt_resized * prompt_weight

        return image
    
class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        ctx_init = cfg.ctx_init
        if ctx_init is not None:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
        else:
            n_ctx = 16
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.tp = cfg.tp
        
        self.hidden_size = clip_model.visual.conv1.weight.shape[0]  # visual encoder hiden size(768)

        self.ctx = None
        if self.tp:
            if ctx_init:  # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                prompt = build_clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                self.ctx = nn.Parameter(ctx_vectors)
                # nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = ctx_init
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            self.ctx = nn.Parameter(ctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of target model context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([build_clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

        self.device = torch.device("cuda:{}".format(int(cfg.gpu_id)))
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # CLS / SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

        self.dim = clip_model.text_projection.shape[1]

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts