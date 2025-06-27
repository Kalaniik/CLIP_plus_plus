import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder_Trans(nn.Module):
    def __init__(self, cfg, clip_model, prompt_learner=None):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post

        if clip_model.visual.proj is not None:
            self.proj = clip_model.visual.proj
        
        self.vp = cfg.vp
        self.location = cfg.location
        
        dropout = cfg.prompt_dropout if cfg.prompt_dropout is not None else 0.0
        self.prompt_dropout = nn.Dropout(dropout)

        self.prompt_learner = prompt_learner

        self.dim = clip_model.text_projection.shape[1]

    def forward(self, x, vctx=None, return_feat=None):
        # x [B, 3, 224, 224]

        x = self.conv1(x)  # shape = [*, width, grid, grid] = [B, 768, 14, 14]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] = [B, 196, 768]
        x = torch.cat(  # shape = [*, grid ** 2 + 1, width]
                [
                    self.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
                ], 
                dim=1,
            )  
        x = x + self.positional_embedding.to(x.dtype)   # image embedding [B, 197, 768]

        if self.vp and vctx != None:
            x = self.incorporate_prompt(x, vctx)    # [B, 197+num_token, 768]

        x = self.ln_pre(x)      # [B, 197+num_token, 768]

        x = x.permute(1, 0, 2)  # NLD -> LND [197+num_token, B, 768]

        x = self.transformer(x) # [197+num_token, B, 768]

        x = x.permute(1, 0, 2)  # LND -> NLD    [B, 197+num_tokens, 768]
        # data = data.permute(0, 2, 1, 3)
        
        if return_feat:
            Fs = x
            Fs = F.adaptive_avg_pool1d(Fs, self.dim) # [B, 197+num, 512]
        x = self.ln_post(x[:, 0, :])    # [B, 768]
        # data = data[:, :, 0, :]
        # data = self.ln_post(data[:, :, 0, :])  # [12/24, B, 1, 768/1024]

        if self.proj is not None:
            x = x @ self.proj   # [B, 512]
            # data = data @ self.proj

        if return_feat:
            return x, Fs
        
        # return x, data
        return x
    
    def incorporate_prompt(self, x, vctx):
        # combine cdu embeddings with image-patch embeddings

        if self.location == "middle":
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(vctx).expand(x.shape[0], -1, -1).half(),
                    x[:, 1:, :]
                ), dim=1)   # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        else:
            visual_ctx = self.prompt_dropout(vctx).expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        
        return x    # [B, 197 + num_token, 768]
    
class ImageEncoder_Conv(nn.Module):
    def __init__(self, cfg, clip_model, prompt_learner=None):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = clip_model.visual.conv1, clip_model.visual.bn1, clip_model.visual.relu1
        self.conv2, self.bn2, self.relu2 = clip_model.visual.conv2, clip_model.visual.bn2, clip_model.visual.relu2
        self.conv3, self.bn3, self.relu3 = clip_model.visual.conv3, clip_model.visual.bn3, clip_model.visual.relu3
        self.avgpool = clip_model.visual.avgpool

        self.layer1 = clip_model.visual.layer1
        self.layer2 = clip_model.visual.layer2
        self.layer3 = clip_model.visual.layer3
        self.layer4 = clip_model.visual.layer4
        self.attnpool = clip_model.visual.attnpool
        
        self.prompt_learner = prompt_learner

        self.dim = clip_model.text_projection.shape[1]

    def forward(self, x, return_feat=False):
        
        def stem(x):
            for conv, bn, relu in [(self.conv1, self.bn1, self.relu1), (self.conv2, self.bn2, self.relu2), (self.conv3, self.bn3, self.relu3)]:
                x = relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, C, H, W] = [B, 2048, 7, 7]
        if return_feat:     # you can modify the code as you need 
            Fs = x.permute(0, 2, 3, 1).view(x.shape[0], -1, x.shape[1]) # [B, 49, 2048]
            Fs = F.adaptive_avg_pool1d(Fs, self.dim) # [B, 49, 1024]
        x = self.attnpool(x)    # [B, 1024]

        if return_feat:
            return x, Fs
        
        return x

class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
        self.tp = cfg.tp
        
        dropout = cfg.prompt_dropout if cfg.prompt_dropout is not None else 0.0
        self.prompt_dropout = nn.Dropout(dropout)
       
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x