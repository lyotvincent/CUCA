import os, json
import random, yaml
import numpy as np
import argparse
from ast import literal_eval

import torch


def get_parser():
    parser = argparse.ArgumentParser(description='Configurations for end2end finetuning')
    parser.add_argument('-c', '--config', type=str, default=None, help='Path to a benchmark config file')
    parser.add_argument('--opts', help='see bench_config yaml file for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert args.config is not None, "Please provide config file for parameters."

    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    if args.opts is not None:
        assert len(args.opts) % 3 == 0, f"Please provide cls_type, key, value in triplets.\n {args.opts}"
        for cls_type, key, value in zip(args.opts[0::3], args.opts[1::3], args.opts[2::3]):
            assert cls_type in config, 'Non-existent cls_type: {}'.format(config.keys())
            assert key in config[cls_type], 'Non-existent subkey: {}'.format(config[cls_type].keys())
            try: 
                value = literal_eval(value) 
            except: 
                pass
            config[cls_type][key] = value
    
    return config


def set_seed_torch(gpu=None, seed=1):
    if torch.cuda.is_available() and gpu is not None:
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device = torch.device('cuda:'+gpu)
    else:
        device = torch.device('cpu')

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    
    return device


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_trainable_parameters(model: torch.nn) -> None:
    """Print number of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return f"trainable params: {trainable_params} || all params: {all_param}\
          || trainable%: {100 * trainable_params / all_param:.2f}"


def merge_fold_results(arr):
    aggr_dict = {}
    for dict in arr:
        for item in dict['pearson_corrs']:
            gene_name = item['name']
            correlation = item['pearson_corr']
            aggr_dict[gene_name] = aggr_dict.get(gene_name, []) + [correlation]
    
    aggr_results = []
    all_corrs = []
    for key, value in aggr_dict.items():
        aggr_results.append({
            "name": key,
            "pearson_corrs": value,
            "mean": np.mean(value),
            "std": np.std(value)
        })
        all_corrs += value
        
    mean_per_split = [d['pearson_mean'] for d in arr]    
        
    return {"pearson_corrs": aggr_results, 
            "pearson_mean": np.mean(mean_per_split), 
            "pearson_std": np.std(mean_per_split), 
            "mean_per_split": mean_per_split}

