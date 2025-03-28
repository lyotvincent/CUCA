import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from visualizer import get_local
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    # @get_local('attn')
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        # print(attn.shape)
        # quit()
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        return x



class FeatureExtractor(nn.Module):
    """Some Information about FeatureExtractor"""
    def __init__(self, backbone='resnet101'):
        super(FeatureExtractor, self).__init__()
        backbone = torchvision.models.resnet101(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        # self.backbone = backbone
    def forward(self, x):
        x = self.backbone(x)
        return x

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, backbone='resnet50', learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        backbone = torchvision.models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = num_classes
        self.classifier = nn.Linear(num_filters, num_target_classes)
        # self.valid_acc = torchmetrics.Accuracy()
        self.learning_rate = learning_rate

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.feature_extractor(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        return parser


# class HisToGene(pl.LightningModule):
class HisToGene(nn.Module):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=64):
        super().__init__()
        # self.save_hyperparameters()
        # self.learning_rate = learning_rate
        patch_dim = 3*patch_size*patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2*dim, dropout = dropout, emb_dropout = dropout)

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers, **kwargs):
        B, N, C, H, W = patches.shape
        patches = patches.reshape(B, N, C*H*W) # B, N, C*H*W

        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        
        if 'return_embed' in kwargs and kwargs['return_embed']:
            return h, x
        else:
            return x
    # def training_step(self, batch, batch_idx):        
    #     patch, center, exp = batch
    #     pred = self(patch, center)
    #     loss = F.mse_loss(pred.view_as(exp), exp)
    #     self.log('train_loss', loss)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     patch, center, exp = batch
    #     pred = self(patch, center)
    #     loss = F.mse_loss(pred.view_as(exp), exp)
    #     self.log('valid_loss', loss)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     patch, center, exp = batch
    #     pred = self(patch, center)
    #     loss = F.mse_loss(pred.view_as(exp), exp)
    #     self.log('test_loss', loss)

    # def configure_optimizers(self):
    #     # self.hparams available because we called self.save_hyperparameters()
    #     return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument('--learning_rate', type=float, default=0.0001)
    #     return parser

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    a = torch.rand(1,4000,3*112*112)
    p = torch.ones(1,4000,2).long()
    model = HisToGene()
    print(count_parameters(model))
    x = model(a,p)
    print(x.shape)
