import os
import torch
import torch.optim as optim


from models.Hist2cell_arch import Hist2Cell
from models.THItoGene_arch.vis_model import THItoGene
from models.HIST2ST_arch import Hist2ST
from models.HisToGene_arch import HisToGene
from models.FMMLP_arch import FMMLP, MLP, LinearProbing
from models.CUCA_arch import CUCA, CUCAMLP

from utils.loss_utils import RMSELoss, PearsonLoss, InfoNCE


def _init_optim(model, optim_func=None, lr=1e-4, reg=1e-5, scheduler_func=None, lr_adj_iteration=100):
    if optim_func == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    elif optim_func == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg)
    elif optim_func == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=reg)
    elif optim_func == "radam":
        optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=reg) 

    else:
        raise NotImplementedError

    if scheduler_func == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=20, eta_min=lr*0.1, verbose=True) #设置余弦退火算法调整学习率，每个epoch调整
    elif scheduler_func == "CyclicLR":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                base_lr=lr*0.25, max_lr=lr, step_size_up=lr_adj_iteration//6, 
                                                cycle_momentum=False, verbose=True) #
    elif scheduler_func == "LinearLR":
        scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, 
                                                start_factor=1, end_factor=0.1, total_iters=lr_adj_iteration//2, verbose=True)
    elif scheduler_func == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                                  max_lr=lr, total_steps=lr_adj_iteration, pct_start=0.2, div_factor=10, final_div_factor=10, verbose=True)
    elif scheduler_func == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=50, gamma=0.9, verbose=True)
    elif scheduler_func == "NoAdjust" or scheduler_func is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=lr_adj_iteration, gamma=0.3, verbose=True) # no change for lr
    else:
        raise NotImplementedError
    
    return optimizer, scheduler


def _init_loss_function(loss_func=None):
    r"""
    Init loss function
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss
    """
    if loss_func == 'CE':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss_func == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif loss_func == 'RMSE':
        loss_fn = RMSELoss()
    elif loss_func == 'KL':
        loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    elif loss_func == 'L1':
        loss_fn = torch.nn.L1Loss()
    elif loss_func == 'InfoNCE':
        loss_fn = InfoNCE()
    elif loss_func == 'Pearson':
        loss_fn = PearsonLoss()    
    else:
        raise NotImplementedError  
    return loss_fn


def _init_model(architecture_name, 
                backbone_name="resnet18",
                num_cls=1,
                hidden_dim=256, 
                proj_dim=256, 
                **lora_cfg_kwargs):

    if architecture_name == "hist2cell":
        model = Hist2Cell(backbone=backbone_name, cell_dim=num_cls, vit_depth=3)

    elif architecture_name == "THItoGene":
        model = THItoGene(n_genes=num_cls, route_dim=64, caps=20, heads=[16, 8], n_layers=4, n_pos=128)
    
    elif architecture_name == "HisToGene":
        model = HisToGene(n_layers=8, n_genes=num_cls, patch_size=112, n_pos=128)
    
    elif architecture_name == "Hist2ST":
        model = Hist2ST(n_genes=num_cls, depth1=2, depth2=8, depth3=4, 
                        kernel_size=5, patch_size=7, 
                        fig_size=112, heads=16, channel=32, 
                        dropout=0.2, zinb=0, nb=False, 
                        bake=5, lamb=0.5, policy='mean', n_pos=128) # 240 is for humanlung dataset else 128
    
    elif architecture_name == "ST-Net":
        import torchvision
        model  = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights)
        del model.classifier
        model.add_module('classifier', torch.nn.Linear(1024, num_cls))

    elif architecture_name == "LinearProbing":
        model = LinearProbing(backbone_name, num_cls)

    elif architecture_name == "FMMLP":
        model = FMMLP(backbone_name, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=True, **lora_cfg_kwargs)

    elif architecture_name == "MLP":
        model = MLP(backbone_name, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=True)

    elif architecture_name == "CUCA":
        model = CUCA(backbone_name, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=True, aux_output=250, embed_type="geneexp", **lora_cfg_kwargs)
    elif architecture_name == "CUCAMLP":
        model = CUCAMLP(backbone_name, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=True, aux_output=250, embed_type="geneexp")
    else:
        raise NotImplementedError
    
    return model



if __name__ == "__main__":
    pass