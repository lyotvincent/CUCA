
import os
import torch
import torch.nn as nn

from peft import PeftModel, get_peft_model, LoraConfig
from models.FoundationModels import inf_encoder_factory


class LinearProbing(nn.Module):
    def __init__(self, backbone, num_cls, **kwargs):
        super(LinearProbing, self).__init__()
        backbone_embed_dim_dict = {"hoptimus0": 1536, "gigapath": 1536, 
                                   "virchow": 2560, "virchow2": 2560, 
                                   "uni_v1": 1024, "conch_v1": 512, "plip": 768,
                                   "phikon": 768, "ctranspath": 768,
                                   "resnet50": 512}
        backbone_out_embed_dim = backbone_embed_dim_dict[backbone]

        self.linearprob_head = torch.nn.Linear(backbone_out_embed_dim, num_cls)

    def forward(self, x, **kwargs):
        reg_pred = self.linearprob_head(x)
        
        if 'return_embed' in kwargs and kwargs['return_embed']:
            return x, reg_pred
        else:
            return reg_pred
    

class MLP(nn.Module):
    def __init__(self, backbone, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=False):
        super(MLP, self).__init__()
        backbone_embed_dim_dict = {"hoptimus0": 1536, "gigapath": 1536, 
                                   "virchow": 2560, "virchow2": 2560, 
                                   "uni_v1": 1024, "conch_v1": 512, "plip": 768,
                                   "phikon": 768, "ctranspath": 768,
                                   "resnet50": 512}
        backbone_out_embed_dim = backbone_embed_dim_dict[backbone]

        self.projector_head = nn.Sequential(
            nn.Linear(backbone_out_embed_dim, hidden_dim),  
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            )
        
        self.regression_head = torch.nn.Linear(proj_dim, num_cls)

    def forward(self, x, **kwargs):
        proj_embed = self.projector_head(x)
        reg_pred = self.regression_head(proj_embed)
        
        if 'return_embed' in kwargs and kwargs['return_embed']:
            return proj_embed, reg_pred
        else:
            return reg_pred



class FMMLP(nn.Module):
    def __init__(self, backbone, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=False, **LoraCfgParams):
        super(FMMLP, self).__init__()

        weights_path = os.path.join("model_weights_pretrained", backbone)
        self.backbone = inf_encoder_factory(backbone)(weights_path)

        backbone_out_embed_dim = self.backbone.out_embed_dim

        self.projector_head = nn.Sequential(
            nn.Linear(backbone_out_embed_dim, hidden_dim),  
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            )
        
        self.regression_head = torch.nn.Linear(proj_dim, num_cls)
        self.process(**LoraCfgParams)

    def forward(self, x, **kwargs):
        embedding = self.backbone(x)
        proj_embed = self.projector_head(embedding)
        reg_pred = self.regression_head(proj_embed)
        
        return reg_pred


    def process(self, **lora_cfg_kwargs):
        if lora_cfg_kwargs['ft_lora']:
            del lora_cfg_kwargs['ft_lora']
            for name, module in self.backbone.encoder.named_modules(): # get the target modules in encoder
                if isinstance(module, torch.nn.Linear) and name.split('.')[1] in lora_cfg_kwargs['only_spec_blocks']:
                    lora_cfg_kwargs['target_modules'].append(name)
            del lora_cfg_kwargs['only_spec_blocks']

            lora_config = LoraConfig(**lora_cfg_kwargs)
            self.backbone.encoder = get_peft_model(self.backbone.encoder, lora_config) 
        else: # no lora, fixed
            for name, param in self.backbone.encoder.named_parameters():
                param.requires_grad = False


if __name__ == "__main__":
    pass