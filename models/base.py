import torch
import torch.nn as nn


class Base_PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        pass

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        '''
        dim0 is either batch_size (during training) or n_cls (during testing)
        ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        '''
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]


        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx.half(),  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        if hasattr(self, 'dropout') and self.dropout != None:
            prompts = self.dropout(prompts)
            
        return prompts

    def forward(self):
        pass