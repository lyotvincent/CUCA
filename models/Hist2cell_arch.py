import os

import torch
import torch.nn as nn

import torchvision.models as tvmodels

from torch_geometric.nn import GATv2Conv, LayerNorm

from models.ViT import Mlp, VisionTransformer
from models.FoundationModels import inf_encoder_factory


class Hist2Cell(nn.Module):
    def __init__(self, cell_dim=80, vit_depth=3, backbone='resnet18'):
        super(Hist2Cell, self).__init__()
        if backbone == 'resnet18':
            self.backbone = tvmodels.resnet18(weights=tvmodels.ResNet18_Weights.IMAGENET1K_V1)
            backbone_out_embed_dim = self.backbone.inplanes
            self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])

        else:
            weights_path = os.path.join("model_weights_pretrained", backbone)
            self.backbone = inf_encoder_factory(backbone)(weights_path)
            backbone_out_embed_dim = self.backbone.out_embed_dim

            for name, param in self.backbone.encoder.named_parameters():
                param.requires_grad = False        


        self.embed_dim = 32 * 8
        self.head = 8
        self.dropout = 0.3
        
        self.conv1 = GATv2Conv(in_channels=backbone_out_embed_dim, out_channels=int(self.embed_dim/self.head), heads=self.head)
        self.norm1 = LayerNorm(in_channels=self.embed_dim)
        
        self.cell_transformer = VisionTransformer(num_classes=cell_dim, embed_dim=self.embed_dim, depth=vit_depth,
                                                  mlp_head=True, drop_rate=self.dropout, attn_drop_rate=self.dropout)
        self.spot_fc = nn.Linear(in_features=backbone_out_embed_dim, out_features=256)
        self.spot_head = Mlp(in_features=256, hidden_features=512*2, out_features=cell_dim)
        self.local_head = Mlp(in_features=256, hidden_features=512*2, out_features=cell_dim)
        self.fused_head = Mlp(in_features=256, hidden_features=512*2, out_features=cell_dim)
    
    
    def forward(self, x, edge_index, return_embed=False):
        x_spot = self.backbone(x)
        x_spot = x_spot.squeeze()
        
        x_local = self.conv1(x=x_spot, edge_index=edge_index)
        x_local = self.norm1(x_local)
        
        x_local = x_local.unsqueeze(0)
        
        x_cell = x_local
        
        x_spot = self.spot_fc(x_spot)
        cell_predication_spot = self.spot_head(x_spot)
        x_local = x_local.squeeze(0)
        cell_prediction_local = self.local_head(x_local)
        cell_prediction_global, x_global = self.cell_transformer(x_cell)
        cell_prediction_global = cell_prediction_global.squeeze()
        x_global = x_global.squeeze()

        fused_embedd = (x_spot+x_local+x_global)/3.0
        cell_prediction_fused = self.fused_head(fused_embedd)
        cell_prediction = (cell_predication_spot + cell_prediction_local + cell_prediction_global + cell_prediction_fused) / 4.0
        
        if return_embed:
            return fused_embedd, cell_prediction
        else:
            return cell_prediction

if __name__ == "__main__":
    pass