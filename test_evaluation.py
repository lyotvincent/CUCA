import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import argparse, json
from scipy.stats import pearsonr

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import NeighborLoader

from utils.dataset_utils import ImgCellGeneDataset, EmbedCellGeneDataset, THItoGeneDataset, load_graph_pt_data
from utils.general_utils import get_parser, set_seed_torch
from utils.init_utils import _init_optim, _init_loss_function, _init_model

from models.FoundationModels import inf_encoder_factory


@torch.inference_mode()
def external_eval(cur_split, test_loader, exp_res_dir=None, device="cuda", **param_kwargs):
    # fm_list = ["hoptimus0", "gigapath", "virchow2", "virchow", "uni_v1", "phikon", "plip", "conch_v1", "resnet50"]
    # if param_kwargs['backbone'] in fm_list:
    #     embed_path = os.path.join(os.path.dirname(exp_res_dir), 'data_embeds')
    #     embed_filename = os.path.join(embed_path, f"{param_kwargs['backbone']}_fold_{cur_split}_test.h5")
        
    #     if os.path.exists(embed_filename) and False:
    #         test_loader = torch.utils.data.DataLoader(EmbedCellGeneDataset(embed_filename), shuffle=False, 
    #                                                   batch_size=test_loader.batch_size, 
    #                                                   num_workers=test_loader.num_workers)   

    if param_kwargs['architecture'] in ["LinearProbing", "MLP", "CUCAMLP"]:
        print(f"Initialing {param_kwargs['backbone']} backbone encoder...")      

        weights_path = os.path.join("model_weights_pretrained", param_kwargs['backbone'])
        encoder = inf_encoder_factory(param_kwargs['backbone'])(weights_path)
        encoder.eval()
        encoder.to(device)
    else:
        pass

    model = _init_model(architecture_name=param_kwargs['architecture'], 
                        backbone_name=param_kwargs['backbone'],
                        num_cls=param_kwargs['num_cls'], 
                        hidden_dim=param_kwargs['hidden_dim'],
                        proj_dim=param_kwargs['proj_dim'], 
                        **param_kwargs['LoraCfgParams']
                        )
    checkpoint = torch.load(os.path.join(exp_res_dir, f"split_{cur_split}", f"ckpt_{cur_split}.pth"), map_location='cpu')
    # checkpoint = {k.replace('resnet18.', 'backbone.'): v for k, v in checkpoint.items()}   
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    criterion = _init_loss_function(loss_func=param_kwargs['loss_main'])
    criterion = criterion.to(device)

    test_embed_array = []
    test_sample_num = 0
    test_cell_pos_array = []
    test_cell_pred_array = []
    test_cell_label_array = []
    test_cell_abundance_loss = 0
    gene_cell_split_idx = 250
    for graph in tqdm(test_loader):
        x = graph['x'].to(device)
        gene_exp_cell_abd_label = graph['y'].to(device)
        pos_centers = graph['pos'].to(device)

        edge_index = graph['edge_index'].to(device)
        cell_label = gene_exp_cell_abd_label[..., gene_cell_split_idx:]
        
        if param_kwargs['architecture'] in ["LinearProbing", "MLP", "CUCAMLP"]:
            x = encoder(x)

        if param_kwargs['architecture'] in ["THItoGene", "HisToGene", "Hist2ST"]:
            if param_kwargs['architecture'] == "Hist2ST":
                pred_outputs, _, _ = model(patches=x, centers=graph['pos'].to(device).long(), adj=graph['adj'].to(device))
            else:
                pred_outputs = model(patches=x, centers=graph['pos'].to(device), adj=graph['adj'].to(device))
            pred_outputs = pred_outputs.squeeze(0) # remove the batch (slide) dimension
            cell_label = cell_label.squeeze(0) # remove the batch (slide) dimension
            gene_exp_cell_abd_label = gene_exp_cell_abd_label.squeeze(0) # remove the batch (slide) dimension

        elif param_kwargs['architecture'] in ["ST-Net"]:
            pred_outputs = model(x=x)

        else:
            proj_embed, pred_outputs = model(x=x, edge_index=edge_index, return_embed=True)

        if isinstance(criterion, torch.nn.KLDivLoss): # KL divergence loss requires log_softmax
            cell_loss = criterion(torch.nn.functional.log_softmax(pred_outputs, dim=1), 
                                torch.nn.functional.log_softmax(cell_label, dim=1))        
        else:
            cell_loss = criterion(pred_outputs, cell_label)        

        center_num = len(graph['input_id']) if 'input_id' in graph else cell_label.shape[0] # get the batch size
        center_cell_label = gene_exp_cell_abd_label[:center_num, :]
        center_cell_pred = pred_outputs[:center_num, :]
        center_cell_pos = pos_centers[:center_num, :]

        test_embed_array.append(proj_embed[:center_num, :].squeeze().cpu().numpy())
        test_cell_label_array.append(center_cell_label.squeeze().cpu().detach().numpy())
        test_cell_pred_array.append(center_cell_pred.squeeze().cpu().detach().numpy())
        test_cell_pos_array.append(center_cell_pos.squeeze().cpu().detach().numpy())
        test_sample_num = test_sample_num + center_num
        
        test_cell_abundance_loss += cell_loss.item() * center_num
        
    test_cell_abundance_loss = test_cell_abundance_loss / test_sample_num

    if len(test_cell_pred_array[-1].shape) == 1:
        test_cell_pred_array[-1] = np.expand_dims(test_cell_pred_array[-1], axis=0)
    test_cell_pred_array = np.concatenate(test_cell_pred_array)
    if len(test_cell_label_array[-1].shape) == 1:
        test_cell_label_array[-1] = np.expand_dims(test_cell_label_array[-1], axis=0)
    test_cell_label_array = np.concatenate(test_cell_label_array)

    if len(test_cell_pos_array[-1].shape) == 1:
        test_cell_pos_array[-1] = np.expand_dims(test_cell_pos_array[-1], axis=0)
    test_cell_pos_array = np.concatenate(test_cell_pos_array)

    test_embed_array = np.concatenate(test_embed_array)

    dict_split_cell_abundance_pearson = {}
    for cell_idx in range(test_cell_pred_array.shape[1]):
        r, p = pearsonr(test_cell_pred_array[:, cell_idx], test_cell_label_array[:, gene_cell_split_idx+cell_idx])
        dict_split_cell_abundance_pearson.update({f"celltype_{cell_idx}": {"pcc": r, "pval": p}})

    dict_slides_cell_abundance_pearson = {}
    dict_slides_spot_Predictions = dict()

    test_dataset = test_loader.data if hasattr(test_loader, 'data') else test_loader.dataset # get the dataset object from the loader
    for slide_no in range(test_dataset.batch_size):
        indices = np.where(test_dataset.batch.numpy() == slide_no)
        test_cell_pred_array_sub = test_cell_pred_array[indices]
        test_cell_label_array_sub = test_cell_label_array[indices]
        test_cell_pos_array_sub = test_cell_pos_array[indices]
        test_embed_array_sub = test_embed_array[indices]
        
        dict_slides_spot_Predictions[f"id_{slide_no}"] = {'cell_abundance_predictions': test_cell_pred_array_sub,
                                         'cell_abundance_labels': test_cell_label_array_sub,
                                         'coords': test_cell_pos_array_sub,
                                         'embeds': test_embed_array_sub
                                         }

        dict_one_slide_all_celltype_pearson = {}
        for cell_idx in range(test_cell_pred_array_sub.shape[1]):
            r, p = pearsonr(test_cell_pred_array_sub[:, cell_idx], test_cell_label_array_sub[:, gene_cell_split_idx+cell_idx])
            dict_one_slide_all_celltype_pearson.update({f"celltype_{cell_idx}": {"pcc": r, "pval": p}})

        dict_slides_cell_abundance_pearson.update({f"id_{slide_no}": pd.DataFrame(dict_one_slide_all_celltype_pearson)})

    return dict_split_cell_abundance_pearson, dict_slides_cell_abundance_pearson, dict_slides_spot_Predictions, test_cell_abundance_loss




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test evaluation')
    parser.add_argument('-ep', '--exp_path', type=str, default=None, help='Path to a experiment config file')
    args = parser.parse_args()

    with open(os.path.join(args.exp_path, "configs.json"), 'r') as f:
        config = json.load(f)

    device = set_seed_torch(**config["COMMON"])

    with open(os.path.join(config["CKPTS"]["data_root"], 'cell_types.pkl'), "rb") as f:
        cell_type_info = pickle.load(f)
    cell_type_info = {f"celltype_{idx}": cell_type_info[idx] for idx in range(len(cell_type_info))}
    
    all_splits_pearson = {}
    all_slides_pearson = {}
    all_slides_predictions = {}
    all_splits_loss = {}
    
    hop = 2
    subgraph_bs = config["HyperParams"]['batch_size']
    num_workers = config["HyperParams"]['num_workers']   

    dataset_name = config["CKPTS"]["data_root"].split('/')[-1]
                                                                                                                        
    for cur_split in config["CKPTS"]['split_ids']:
        spec_name = "fold" if dataset_name != "humanlung_cell2location" else "leave"
        split_file_name=os.path.join(config["CKPTS"]["split_data_root"], f"test_{spec_name}_{cur_split}.txt")
        test_slides = open(split_file_name).read().split('\n')
        print(f"evaluating fold {cur_split} with {len(test_slides)} slides:\n {test_slides}")

        if config["HyperParams"]["architecture"] == "hist2cell":           
            test_dataset = load_graph_pt_data(split_file_name=split_file_name, data_root=config["CKPTS"]["data_root"])

            test_loader = NeighborLoader(
                        test_dataset,
                        num_neighbors=[-1]*hop,
                        batch_size=subgraph_bs,
                        directed=False,
                        input_nodes=None,
                        shuffle=False,
                        num_workers=num_workers,
                    )

        elif config["HyperParams"]["architecture"] in ["THItoGene", "HisToGene", "Hist2ST"]:
            test_dataset = THItoGeneDataset(split_file_name=split_file_name, data_root=config["CKPTS"]["data_root"])
            test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=num_workers)

        elif config["HyperParams"]["architecture"] in ["LinearProbing", "FMMLP", "MLP", "CUCA", "CUCAMLP", "ST-Net"]:
            test_dataset = ImgCellGeneDataset(split_file_name=split_file_name, data_root=config["CKPTS"]["data_root"])      
            test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=subgraph_bs, num_workers=num_workers)

        else:
            raise NotImplementedError        

        config['HyperParams']['LoraCfgParams'] = config['LoraCfgParams']

        dict_split_cell_type_pcc, dict_slides_cell_type_pcc, dict_slides_spot_predictions, test_loss = external_eval(cur_split, test_loader, args.exp_path, device=device, **config["HyperParams"])

        all_splits_pearson.update({cur_split: pd.DataFrame(dict_split_cell_type_pcc).rename(columns=cell_type_info)})
        all_splits_loss.update({cur_split: test_loss})

        slide_name_mapping = {f"id_{idx}": test_slides[idx] for idx in range(len(test_slides))}
        dict_slides_cell_type_pcc = {slide_name_mapping[slide_id]: dict_slides_cell_type_pcc[slide_id] for slide_id in dict_slides_cell_type_pcc.keys()}
        dict_slides_spot_predictions = {slide_name_mapping[slide_id]: dict_slides_spot_predictions[slide_id] for slide_id in dict_slides_spot_predictions.keys()}

        all_slides_pearson.update(dict_slides_cell_type_pcc)
        all_slides_predictions.update(dict_slides_spot_predictions)

    all_splits_pearson = pd.concat(all_splits_pearson.values(), keys=[name for name in all_splits_pearson.keys()])

    all_splits_pearson.loc["split_mean"] = (all_splits_pearson.iloc[0::2, :].mean(0)) # mean along all splits
    all_splits_pearson.insert(all_splits_pearson.shape[1], 'celltype_mean', all_splits_pearson.mean(1).values) # mean along all cell types

    all_splits_pearson.to_csv(os.path.join(args.exp_path, "all_splits_all_celltypes_pearson_embed.csv"))
    print(f"all_splits_pearson: {all_splits_pearson}")
    
    all_slides_pearson = pd.concat(all_slides_pearson.values(), keys=[name for name in all_slides_pearson.keys()])
    all_slides_pearson = all_slides_pearson.rename(columns=cell_type_info)
    
    all_slides_pearson.loc["slide_mean"] = (all_slides_pearson.iloc[0::2, :].mean(0)) # mean along all splits
    all_slides_pearson.insert(all_slides_pearson.shape[1], 'celltype_mean', all_slides_pearson.mean(1).values) # mean along all cell types

    all_slides_pearson.to_csv(os.path.join(args.exp_path, "all_slides_all_celltypes_pearson_embed.csv"))
    print(f"all_slides_pearson: {all_slides_pearson}")

    print(f"all_splits_loss: {all_splits_loss}")

    with open(os.path.join(args.exp_path, "all_slides_test_spot_predictions_embed.pkl"), "wb") as f:
        pickle.dump(all_slides_predictions, f)

    # Independent test set
    if "independent_list" in config["CKPTS"] and config["CKPTS"]["independent_list"] is not None:
        for independent_set in config["CKPTS"]["independent_list"]:
            split_file_name=os.path.join(config["CKPTS"]["split_data_root"], independent_set)
            test_slides = open(split_file_name).read().split('\n')

            if config["HyperParams"]["architecture"] == "hist2cell":           
                test_dataset = load_graph_pt_data(split_file_name=split_file_name, data_root=config["CKPTS"]["independent_root"])

                test_loader = NeighborLoader(
                            test_dataset,
                            num_neighbors=[-1]*hop,
                            batch_size=subgraph_bs,
                            directed=False,
                            input_nodes=None,
                            shuffle=False,
                            num_workers=num_workers,
                        )
            elif config["HyperParams"]["architecture"] in ["THItoGene", "HisToGene", "Hist2ST"]:
                split_dataset = THItoGeneDataset(split_file_name=split_file_name, data_root=config["CKPTS"]["independent_root"])
                split_loader = torch.utils.data.DataLoader(split_dataset, shuffle=False, batch_size=1, num_workers=num_workers)

            elif config["HyperParams"]["architecture"] in ["LinearProbing", "MLP", "CUCAMLP", "FMMLP", "CUCA", "ST-Net"]:
                test_dataset = ImgCellGeneDataset(split_file_name=split_file_name, data_root=config["CKPTS"]["independent_root"])      
                test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=subgraph_bs, num_workers=num_workers)

            else:
                raise NotImplementedError        

            config['HyperParams']['LoraCfgParams'] = config['LoraCfgParams']

            all_splits_pearson = {}
            all_slides_pearson = {}
            all_slides_predictions = {}
            all_splits_loss = {}
            
            for cur_split in config["CKPTS"]['split_ids']:
                print(f"Model_fold{cur_split} is evaluating independent_set {independent_set} of {len(test_slides)} slides:\n {test_slides}")

                dict_split_cell_type_pcc, dict_slides_cell_type_pcc, dict_slides_spot_predictions, test_loss = external_eval(cur_split, test_loader, args.exp_path, device=device, **config["HyperParams"])

                all_splits_pearson.update({cur_split: pd.DataFrame(dict_split_cell_type_pcc).rename(columns=cell_type_info)})
                all_splits_loss.update({cur_split: test_loss})

                slide_name_mapping = {f"id_{idx}": f"fold{cur_split}_{test_slides[idx]}" for idx in range(len(test_slides))}
                dict_slides_cell_type_pcc = {slide_name_mapping[slide_id]: dict_slides_cell_type_pcc[slide_id] for slide_id in dict_slides_cell_type_pcc.keys()}
                all_slides_pearson.update(dict_slides_cell_type_pcc)

                dict_slides_spot_predictions = {slide_name_mapping[slide_id]: dict_slides_spot_predictions[slide_id] for slide_id in dict_slides_spot_predictions.keys()}
                all_slides_predictions.update(dict_slides_spot_predictions)

            all_splits_pearson = pd.concat(all_splits_pearson.values(), keys=[name for name in all_splits_pearson.keys()])

            all_splits_pearson.loc["split_mean"] = (all_splits_pearson.iloc[0::2, :].mean(0)) # mean along all splits
            all_splits_pearson.insert(all_splits_pearson.shape[1], 'celltype_mean', all_splits_pearson.mean(1).values) # mean along all cell types

            all_splits_pearson.to_csv(os.path.join(args.exp_path, f"all_splits_all_celltypes_pearson_{independent_set}.csv"))
            print(f"all_splits_pearson: {all_splits_pearson}")
            
            all_slides_pearson = pd.concat(all_slides_pearson.values(), keys=[name for name in all_slides_pearson.keys()])
            all_slides_pearson = all_slides_pearson.rename(columns=cell_type_info)
            
            all_slides_pearson.loc["slide_mean"] = (all_slides_pearson.iloc[0::2, :].mean(0)) # mean along all splits
            all_slides_pearson.insert(all_slides_pearson.shape[1], 'celltype_mean', all_slides_pearson.mean(1).values) # mean along all cell types

            all_slides_pearson.to_csv(os.path.join(args.exp_path, f"all_slides_all_celltypes_pearson_{independent_set}.csv"))
            print(f"all_slides_pearson: {all_slides_pearson}")

            print(f"all_splits_loss: {all_splits_loss}")

            with open(os.path.join(args.exp_path, f"all_slides_spot_predictions_{independent_set}.pkl"), "wb") as f:
                pickle.dump(all_slides_predictions, f)