
import os
import torch
import torch.nn as nn

from peft import PeftModel, get_peft_model, LoraConfig
from models.FoundationModels import inf_encoder_factory
from models.genomic_snn import SNN


class CUCAMLP(nn.Module):
    def __init__(self, backbone, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=False, aux_output=250, embed_type=None):
        super(CUCAMLP, self).__init__()

        self.embed_type = embed_type

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

        if self.embed_type == "geneexp":
            self.snn_branch = SNN(input_dim=aux_output, model_size_omic='huge', n_classes=aux_output)
        elif self.embed_type == "genept":
            self.snn_branch = SNN(input_dim=1536, model_size_omic='huge', n_classes=aux_output) # genept embedding dims 1536
        else:
            pass

    def forward(self, x, **kwargs):
        proj_embed = self.projector_head(x)
        reg_pred = self.regression_head(proj_embed)
        
        if self.embed_type in ["geneexp", "genept"]: # gene expression embedding or genePT embedding modes
            if 'gene_exp' in kwargs and 'gene_embed' in kwargs:
                batch_embedding = torch.matmul(kwargs['gene_exp'], kwargs['gene_embed']) if kwargs['gene_embed'] is not None else kwargs['gene_exp']

                molecu_embed, reconstr_pred = self.snn_branch(batch_embedding)
                return proj_embed, reg_pred, molecu_embed, reconstr_pred
        
        if 'return_embed' in kwargs and kwargs['return_embed']:
            return proj_embed, reg_pred
        else:
            return reg_pred # no embedding or commonly when inference



class CUCA(nn.Module):
    r"""
    CUCA model
    Args:
        - backbone : str
            The name of the backbone model
        - num_cls : int
            The number of classes
        - hidden_dim : int
            The hidden dimension of the projector head
        - proj_dim : int
            The projection dimension
        - dropout : float
            The dropout rate
        - batch_norm : bool
            Whether to use batch normalization
        - embed_type : str
            The type of embedding, one of ["geneexp", "genept", "no"]
        - **LoraCfgParams : dict
            The parameters for PE
    """
    def __init__(self, backbone, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=False, aux_output=250, embed_type=None, **LoraCfgParams):
        super(CUCA, self).__init__()

        self.embed_type = embed_type

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
            nn.BatchNorm1d(proj_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  
            )
        
        self.regression_head = torch.nn.Linear(proj_dim, num_cls)
        self.process_backbone(**LoraCfgParams)

        if self.embed_type == "geneexp":
            self.snn_branch = SNN(input_dim=aux_output, model_size_omic='huge', n_classes=aux_output)
        elif self.embed_type == "genept":
            self.snn_branch = SNN(input_dim=1536, model_size_omic='huge', n_classes=aux_output) # genept embedding dims 1536
        else:
            pass

    def process_backbone(self, **lora_cfg_kwargs):
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

    def forward(self, x, **kwargs):
        embedding = self.backbone(x)
        proj_embed = self.projector_head(embedding)
        reg_pred = self.regression_head(proj_embed)
        
        if self.embed_type in ["geneexp", "genept"]: # gene expression embedding or genePT embedding modes
            if 'gene_exp' in kwargs and 'gene_embed' in kwargs:
                batch_embedding = torch.matmul(kwargs['gene_exp'], kwargs['gene_embed']) if kwargs['gene_embed'] is not None else kwargs['gene_exp']

                molecu_embed, reconstr_pred = self.snn_branch(batch_embedding)
                return proj_embed, reg_pred, molecu_embed, reconstr_pred
        
        if 'return_embed' in kwargs and kwargs['return_embed']:
            return proj_embed, reg_pred
        else:
            return reg_pred # no embedding or commonly when inference


if __name__ == "__main__":
    pass