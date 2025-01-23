import torch
from torch import nn
import math


# borrowed from https://github.com/mahmoodlab/MCAT/

##########################
#### Genomic FC Model ####
##########################
class SNN(nn.Module):
    def __init__(self, input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(SNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256], 
                               'medium': [512, 512, 512], 
                               'big': [1024, 1024, 256],
                               'huge': [1024, 1024, 512]}        
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [Reg_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:-1]):
            fc_omic.append(Reg_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        fc_omic.append(nn.Linear(hidden[-2], hidden[-1]))
        
        self.fc_omic = nn.Sequential(*fc_omic)

        self.classifier = Decoder_Block(hidden[-1], hidden[-1], n_classes, dropout=0.25, batch_norm=False)
        # init_max_weights(self)
        initialize_weights(self)

    def forward(self, x):
        features = self.fc_omic(x)

        logits = self.classifier(features)
        return features, logits

    def relocate(self):
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
            else:
                self.fc_omic = self.fc_omic.to(device)


            self.classifier = self.classifier.to(device)


def Reg_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block (Linear + ReLU + Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))


def Decoder_Block(input_dim, hidden_dim, output_dim, dropout=0.25, batch_norm=False):
    r"""
    Multilayer Reception Block with linear + ReLU + Dropout + Linear

    args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden features
        output_dim (int): Dimension of output features
        dropout (float): Dropout rate,              [default=0.25]
        batch_norm (bool): Batch normalization,     [default=False]
    """
    block_head = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),  
        nn.ReLU(),  # 激活函数
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim)  # 输出层，50个基因表达预测
        )
    
    return block_head


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()