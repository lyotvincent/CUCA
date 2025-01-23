import sys,os
import random, time, json
import numpy as np
import pandas as pd

from tqdm import tqdm

from loguru import logger

from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import NeighborLoader
import torch_geometric
torch_geometric.typing.WITH_PYG_LIB = False


cur_main_dir = os.path.dirname(os.path.abspath(__file__)) # current file path to sys.path
print(cur_main_dir)
sys.path.append(cur_main_dir) 
os.chdir(cur_main_dir)

from models.FoundationModels import inf_encoder_factory
from utils.dataset_utils import ImgCellGeneDataset, EmbedCellGeneDataset, THItoGeneDataset, load_graph_pt_data
from utils.general_utils import get_parser, set_seed_torch, AverageMeter
from utils.file_utils import save_hdf5
from utils.init_utils import _init_optim, _init_loss_function, _init_model



@torch.inference_mode()
def embed_tiles(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    embedding_save_path: str,
    device: str,
    precision
):
    """ Extract embeddings from tiles using `encoder` and save to an h5 file (TODO move to hestcore) """
    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs = batch['x'].to(device).float()
        with torch.autocast(device.type, dtype=precision):
            embeddings = model(imgs)
        if batch_idx == 0:
            mode = 'w'
        else:
            mode = 'a'
        asset_dict = {'embeddings': embeddings.cpu().numpy()}
        asset_dict.update({key: np.array(val) for key, val in batch.items() if key != 'x'})
        save_hdf5(embedding_save_path,
                  asset_dict=asset_dict,
                  mode=mode)
    return embedding_save_path


def train_one_epoch(epoch, num_epochs, model, train_loader, optimizer, criterions, writer=None, device="cuda"):
    model.train()

    loss_meter = AverageMeter()

    loss_meter_pred = AverageMeter()
    loss_meter_reconst = AverageMeter()
    loss_meter_alignment = AverageMeter()

    logger.info(f'lr = {optimizer.param_groups[0]["lr"]}')
    
    train_sample_num = 0
    train_cell_pred_array = []
    train_cell_label_array = []
    for batch_idx, graph in enumerate(train_loader):
        x = graph['x'].to(device)
        y = graph['y'].to(device)
        edge_index = graph['edge_index'].to(device)
        
        gene_exp_label = y[..., :250]
        cell_label = y[..., 250:]
        
        if model.__class__.__name__ in ["Hist2Cell", "LinearProbing", "MLP", "FMMLP"]:
            pred_outputs = model(x=x, edge_index=edge_index)

            cell_loss = criterions['criterion_main'](pred_outputs, cell_label)               
            loss_pred = loss_reconst = loss_align = torch.tensor(0.0)
        
        elif model.__class__.__name__ in ["DenseNet"]:
            pred_outputs = model(x=x)
            cell_loss = criterions['criterion_main'](pred_outputs, cell_label)               
            loss_pred = loss_reconst = loss_align = torch.tensor(0.0)
        
        elif model.__class__.__name__ in ["THItoGene", "HisToGene"]:
            pred_outputs = model(patches=x, centers=graph['pos'].to(device).long(), adj=graph['adj'].to(device))
            pred_outputs = pred_outputs.squeeze(0) # remove the batch (slide) dimension
            cell_label = cell_label.squeeze(0) # remove the batch (slide) dimension
            cell_loss = criterions['criterion_main'](pred_outputs, cell_label)               
            loss_pred = loss_reconst = loss_align = torch.tensor(0.0)
        
        elif model.__class__.__name__ in ["Hist2ST"]:
            pred_outputs, _, _ = model(patches=x, centers=graph['pos'].to(device).long(), adj=graph['adj'].to(device))
            pred_outputs = pred_outputs.squeeze(0) # remove the batch (slide) dimension
            cell_label = cell_label.squeeze(0) # remove the batch (slide) dimension
            loss_pred = criterions['criterion_main'](pred_outputs, cell_label)               

            # new_pred_outputs = model.distillation(model.aug(patch=x, center=graph['pos'].to(device).long(), adj=graph['adj'].to(device)))
            # loss_reconst = criterions['criterion_rec'](new_pred_outputs, pred_outputs)

            loss_reconst = loss_align = torch.tensor(0.0)
            cell_loss = criterions['lambda_main']*loss_pred + criterions['lambda_rec']*loss_reconst

        elif model.__class__.__name__ in ["CUCA", "CUCAMLP"]:
            img_embed, pred_outputs, molecu_embed, rec_outputs = model(x=x, gene_exp=gene_exp_label, gene_embed=None)

            loss_pred = criterions['criterion_main'](pred_outputs, cell_label)
            loss_reconst = criterions['criterion_rec'](rec_outputs, gene_exp_label)

            if isinstance(criterions['criterion_align'], torch.nn.KLDivLoss): # KL divergence loss requires log_softmax
                img_embed = torch.nn.functional.log_softmax(img_embed, dim=1)
                molecu_embed = torch.nn.functional.log_softmax(molecu_embed, dim=1)
            loss_align = criterions['criterion_align'](img_embed, molecu_embed)

            # TODO: add cosine similarity loss for alignment
            # loss_align = 1 - torch.nn.functional.cosine_similarity(img_embed, molecu_embed, dim=1).mean()                
            # loss_align = torch.norm(img_embed - molecu_embed, p=2, dim=1)
            cell_loss = criterions['lambda_main']*loss_pred + criterions['lambda_rec']*loss_reconst + criterions['lambda_align']*loss_align

        else:
            raise NotImplementedError

        optimizer.zero_grad()
        cell_loss.backward()
        optimizer.step()

        center_num = len(graph['input_id']) if 'input_id' in graph else cell_label.shape[0] # get the batch size
        loss_meter.update(cell_loss.item(), center_num)
        loss_meter_pred.update(loss_pred.item(), center_num)
        loss_meter_reconst.update(loss_reconst.item(), center_num)
        loss_meter_alignment.update(loss_align.item(), center_num)

        logging_step = len(train_loader) if model.__class__.__name__ in ["THItoGene", "HisToGene", "Hist2ST"] else 20
        if batch_idx % (len(train_loader)//logging_step) == 0:
            logger.info(f'****Epoch/Iter****[{(epoch+1):03d}/{num_epochs}][{(batch_idx + 1):03d}/{len(train_loader)}]'
                        f'****IterationLoss****{loss_meter.val:.4f}.')
            logger.info(f'****Pred | Rec | Align )****[{loss_meter_pred.val:.4f} | {loss_meter_reconst.val:.4f} | {loss_meter_alignment.val:.4f}]')
            
            if writer is not None:
                writer.add_scalar('train/loss', loss_meter.val, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train/loss_pred', loss_meter_pred.val, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train/loss_reconst', loss_meter_reconst.val, epoch * len(train_loader) + batch_idx)  
                writer.add_scalar('train/loss_align', loss_meter_alignment.val, epoch * len(train_loader) + batch_idx)  

        center_cell_label = cell_label[:center_num, :]
        center_cell_pred = pred_outputs[:center_num, :]
        train_cell_label_array.append(center_cell_label.squeeze().cpu().detach().numpy())
        train_cell_pred_array.append(center_cell_pred.squeeze().cpu().detach().numpy())
        train_sample_num = train_sample_num + center_num
    
    if len(train_cell_pred_array[-1].shape) == 1:
        train_cell_pred_array[-1] = np.expand_dims(train_cell_pred_array[-1], axis=0)
    train_cell_pred_array = np.concatenate(train_cell_pred_array)
    if len(train_cell_label_array[-1].shape) == 1:
        train_cell_label_array[-1] = np.expand_dims(train_cell_label_array[-1], axis=0)
    train_cell_label_array = np.concatenate(train_cell_label_array)

    train_cell_abundance_all_pearson_average = 0.0
    for i in range(train_cell_pred_array.shape[1]):
        r, p = pearsonr(train_cell_pred_array[:, i], train_cell_label_array[:, i])
        train_cell_abundance_all_pearson_average = train_cell_abundance_all_pearson_average + r
    train_cell_abundance_all_pearson_average = train_cell_abundance_all_pearson_average / train_cell_pred_array.shape[1]
    return train_cell_abundance_all_pearson_average, loss_meter.avg


@torch.no_grad()
def test_eval(model, test_loader, criterion=None, device='cuda'):
    model.eval()

    test_sample_num = 0
    test_cell_pred_array = []
    test_cell_label_array = []
    test_cell_abundance_loss = 0
    for graph in tqdm(test_loader):
        x = graph['x'].to(device)
        y = graph['y'].to(device)
        edge_index = graph['edge_index'].to(device)
        cell_label = y[..., 250:]
        
        if model.__class__.__name__ in ["THItoGene", "HisToGene", "Hist2ST"]:
            if model.__class__.__name__ == "Hist2ST":
                pred_outputs, _, _ = model(patches=x, centers=graph['pos'].to(device).long(), adj=graph['adj'].to(device))
            else:
                pred_outputs = model(patches=x, centers=graph['pos'].to(device), adj=graph['adj'].to(device))
            pred_outputs = pred_outputs.squeeze(0) # remove the batch (slide) dimension
            cell_label = cell_label.squeeze(0) # remove the batch (slide) dimension

        elif model.__class__.__name__ in ["DenseNet"]:
            pred_outputs = model(x=x)

        else:
            pred_outputs = model(x=x, edge_index=edge_index)

        if criterion is not None:
            if isinstance(criterion, torch.nn.KLDivLoss): # KL divergence loss requires log_softmax
                cell_loss = criterion(torch.nn.functional.log_softmax(pred_outputs, dim=1), 
                                    torch.nn.functional.log_softmax(cell_label, dim=1))        
            else:
                cell_loss = criterion(pred_outputs, cell_label)   
        else:
            cell_loss = torch.tensor(0.0)     
            
        center_num = len(graph['input_id']) if 'input_id' in graph else cell_label.shape[0] # get the batch size
        center_cell_label = cell_label[:center_num, :]
        center_cell_pred = pred_outputs[:center_num, :]
        
        test_cell_label_array.append(center_cell_label.squeeze().cpu().detach().numpy())
        test_cell_pred_array.append(center_cell_pred.squeeze().cpu().detach().numpy())
        test_sample_num = test_sample_num + center_num
        
        test_cell_abundance_loss += cell_loss.item() * center_num
        
    test_cell_abundance_loss = test_cell_abundance_loss / test_sample_num

    if len(test_cell_pred_array[-1].shape) == 1:
        test_cell_pred_array[-1] = np.expand_dims(test_cell_pred_array[-1], axis=0)
    test_cell_pred_array = np.concatenate(test_cell_pred_array)
    if len(test_cell_label_array[-1].shape) == 1:
        test_cell_label_array[-1] = np.expand_dims(test_cell_label_array[-1], axis=0)
    test_cell_label_array = np.concatenate(test_cell_label_array)
        
    dict_test_cell_abundance_all_pearson = {}
    for i in range(test_cell_pred_array.shape[1]):
        if np.isnan(test_cell_pred_array[:, i]).any():
            r, p = -1, -1
        else:
            r, p = pearsonr(test_cell_pred_array[:, i], test_cell_label_array[:, i])
        dict_test_cell_abundance_all_pearson.update({f"celltype_{i}": {"pcc": r, "pval": p}})

    return dict_test_cell_abundance_all_pearson, test_cell_abundance_loss


def main(cur_split, loaders, exp_res_dir=None, device="cuda", **param_kwargs):
    # pre-extracted features
    fm_list = ["hoptimus0", "gigapath", "virchow2", "virchow", "uni_v1", "phikon", "plip", "conch_v1", "resnet50"]
    if param_kwargs['backbone'] in fm_list and param_kwargs['pre_extracted']:
        embed_path = os.path.join(os.path.dirname(exp_res_dir), 'data_embeds')
        os.makedirs(embed_path, exist_ok=True)

        for split_type in loaders.keys():
            embed_filename = os.path.join(embed_path, f"{param_kwargs['backbone']}_fold_{cur_split}_{split_type}.h5")

            if not os.path.exists(embed_filename):
                logger.info(f"Feature pre-extracting for {param_kwargs['backbone']} on {split_type}")      

                weights_path = os.path.join("model_weights_pretrained", param_kwargs['backbone'])
                encoder = inf_encoder_factory(param_kwargs['backbone'])(weights_path)
                _ = encoder.eval()
                encoder.to(device)

                _ = embed_tiles(loaders[split_type], encoder, embedding_save_path=embed_filename, device=device, precision=encoder.precision)
                del encoder
                torch.cuda.empty_cache()

            else:
                logger.info(f"Pre-extracted features exist in {embed_path} for {param_kwargs['backbone']}")
            
            embed_loader = torch.utils.data.DataLoader(EmbedCellGeneDataset(embed_filename), 
                                                    shuffle=split_type=="train", 
                                                    drop_last=True,
                                                    batch_size=loaders[split_type].batch_size, num_workers=loaders[split_type].num_workers)
            loaders.update({split_type: embed_loader})
       
    else:
        logger.info(f"Training from scratch [img] for {param_kwargs['backbone']}")

    model = _init_model(architecture_name=param_kwargs['architecture'], 
                        backbone_name=param_kwargs['backbone'],
                        num_cls=param_kwargs['num_cls'], 
                        hidden_dim=param_kwargs['hidden_dim'],
                        proj_dim=param_kwargs['proj_dim'], 
                        **param_kwargs['LoraCfgParams']
                        )

    model = model.to(device)

    optimizer, scheduler = _init_optim(model, param_kwargs['optim_fn'], param_kwargs['lr_rate'], 
                                    param_kwargs['weight_reg'], 
                                    param_kwargs['scheduler_fn'], lr_adj_iteration=param_kwargs['max_epochs'])
    logger.info(f"******** Init Optimizer: {optimizer}\n Scheduler: {scheduler} ********")

    criterions = {}
    criterion_main = _init_loss_function(loss_func=param_kwargs['loss_main'])
    criterions.update({'criterion_main': criterion_main.to(device), 'lambda_main': param_kwargs['lambda_main']})

    criterion_rec = _init_loss_function(loss_func=param_kwargs['loss_rec'])
    criterions.update({'criterion_rec': criterion_rec.to(device), 'lambda_rec': param_kwargs['lambda_rec']})

    criterion_align = _init_loss_function(loss_func=param_kwargs['loss_align'])
    criterions.update({'criterion_align': criterion_align.to(device), 'lambda_align': param_kwargs['lambda_align']})

    logger.info(f"******** Init Loss Function: {criterions}********\n")


    os.makedirs(os.path.join(exp_res_dir, f"split_{cur_split}"), exist_ok=True)
    writer = SummaryWriter(os.path.join(exp_res_dir, f"split_{cur_split}"), flush_secs=15)

    best_cell_abundance_all_average = -1.0
    since = time.time()
    num_epochs = param_kwargs['max_epochs']
    for epoch in range(num_epochs):
        train_cell_abundance_all_pearson_average, train_loss = train_one_epoch(epoch, num_epochs, 
                                                                               model, loaders['train'], 
                                                                               optimizer, criterions, writer=writer, device=device)        
        scheduler.step()
        if writer is not None:
            writer.add_scalar('train/lr', scheduler.optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('train/pcc', train_cell_abundance_all_pearson_average, epoch)

        time_elapsed = time.time() - since
        logger.info(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
        logger.info(f'Epoch: {(epoch + 1)} \tTraining Cell abundance Loss: {train_loss:.6f}')
        logger.info(f'Epoch: {(epoch + 1)} \tTraining Cell abundance pearson all average: {train_cell_abundance_all_pearson_average:.6f}')

        val_loader = loaders['val'] if 'val' in loaders.keys() else loaders['test'] # 'test' only for humanlung_cell2location
        dict_cell_type_pcc, val_loss = test_eval(model, test_loader=val_loader, criterion=criterions['criterion_main'], device=device)
        val_cell_abundance_all_pearson_average = pd.DataFrame(dict_cell_type_pcc).mean(1).pcc

        if writer is not None:
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/pcc', val_cell_abundance_all_pearson_average, epoch)

        if val_cell_abundance_all_pearson_average > best_cell_abundance_all_average and exp_res_dir is not None:
            best_cell_abundance_all_average = val_cell_abundance_all_pearson_average
            torch.save(model.state_dict(), os.path.join(exp_res_dir, f"split_{cur_split}", f"ckpt_{cur_split}.pth"))
            logger.info(f"saving best cell all abundance average {val_cell_abundance_all_pearson_average}")

        logger.info(f'Epoch: {(epoch + 1)} \tVal Cell abundance Loss: {val_loss:.6f}')
        logger.info(f'Epoch: {(epoch + 1)} \tVal Cell abundance pearson all: {val_cell_abundance_all_pearson_average}\n'
                    f'{pd.DataFrame(dict_cell_type_pcc)}')

    writer.close()

    checkpoint = torch.load(os.path.join(exp_res_dir, f"split_{cur_split}", f"ckpt_{cur_split}.pth"), map_location='cpu')
    # checkpoint = {k.replace('resnet18.', 'backbone.'): v for k, v in checkpoint.items()}   
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    dict_cell_type_pcc, _ = test_eval(model, test_loader=loaders['test'], criterion=None, device=device)
    logger.info(f"Test Cell abundance pearson res. for fold{cur_split}\n: {pd.DataFrame(dict_cell_type_pcc)}")

    test_cell_abundance_all_pearson_average = pd.DataFrame(dict_cell_type_pcc).mean(1).pcc
    return test_cell_abundance_all_pearson_average



if __name__ == "__main__":
    config = get_parser()

    logger.remove()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {message}")
    
    dataset_name = config["CKPTS"]["data_root"].split('/')[-1]
    exp_res_dir = os.path.join(config["CKPTS"]["results_dir"], dataset_name, config["CKPTS"]["exp_code"])
    os.makedirs(exp_res_dir, exist_ok=True)

    logger.add(os.path.join(exp_res_dir, "training.log"), 
               format="{time:YYYY-MM-DD HH:mm:ss} | {message}")

    logger.info(f"Config params: {json.dumps(config, indent = 4)}")
    with open(os.path.join(exp_res_dir, "configs.json"), 'w') as f:
        json.dump(config, f, indent=4) # save the params setting

    device = set_seed_torch(**config["COMMON"])
    
    all_splits_cell_abundance_pearson = []
    for cur_split in config["CKPTS"]['split_ids']:
        logger.info(f"current split: {cur_split}")
        
        hop = 2
        subgraph_bs = config["HyperParams"]['batch_size']
        num_workers = config["HyperParams"]['num_workers']

        loaders = {}
        splits_list = ["train", "val", "test"] if dataset_name != "humanlung_cell2location" else ["train", "test"]
        spec_name = "fold" if dataset_name != "humanlung_cell2location" else "leave"
        for split_type in splits_list:
            if config["HyperParams"]["architecture"] == "hist2cell":
                split_dataset = load_graph_pt_data(split_file_name=os.path.join(config["CKPTS"]["split_data_root"], f"{split_type}_{spec_name}_{cur_split}.txt"),
                                                   data_root=config["CKPTS"]["data_root"])
                split_loader = NeighborLoader(
                    split_dataset, 
                    num_neighbors=[-1]*hop, batch_size=subgraph_bs,
                    directed=False, input_nodes=None,
                    shuffle=split_type=="train", num_workers=num_workers,                )

            elif config["HyperParams"]["architecture"] in ["FMMLP", "LinearProbing", "MLP", "CUCA", "CUCAMLP", "ST-Net"]:
                split_dataset = ImgCellGeneDataset(split_file_name=os.path.join(config["CKPTS"]["split_data_root"], f"{split_type}_{spec_name}_{cur_split}.txt"),
                                                   data_root=config["CKPTS"]["data_root"])
                split_loader = torch.utils.data.DataLoader(split_dataset, shuffle=split_type=="train", 
                                                           drop_last=config["HyperParams"]["architecture"] in ["FMMLP", "CUCA"], 
                                                           batch_size=subgraph_bs, num_workers=num_workers)
            
            elif config["HyperParams"]["architecture"] in ["THItoGene", "HisToGene", "Hist2ST"]:
                split_dataset = THItoGeneDataset(split_file_name=os.path.join(config["CKPTS"]["split_data_root"], f"{split_type}_{spec_name}_{cur_split}.txt"),
                                                   data_root=config["CKPTS"]["data_root"])
                split_loader = torch.utils.data.DataLoader(split_dataset, shuffle=split_type=="train", batch_size=1, num_workers=num_workers)
            else:
                raise NotImplementedError

            loaders.update({split_type: split_loader})

        config['HyperParams']['LoraCfgParams'] = config['LoraCfgParams']
        test_cell_abundance_pearson = main(cur_split, loaders, exp_res_dir=exp_res_dir, device=device, **config["HyperParams"])
        all_splits_cell_abundance_pearson.append(test_cell_abundance_pearson)

    logger.info(f"All folds cell_abundance_pearson on Test cases: {all_splits_cell_abundance_pearson}")

    mean_val = np.array(all_splits_cell_abundance_pearson).mean()
    logger.info(f"Mean cell_abundance_pearson on test cases from all folds: {mean_val}")
    logger.info(f"Training logs saved to: {exp_res_dir}")