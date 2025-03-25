import os
import numpy as np

import torch
import torchvision.transforms as transforms

from PIL import Image
from torch_geometric.data import Batch
from scipy.spatial import distance, KDTree

import scanpy as sc

from utils.file_utils import read_assets_from_h5


class EmbedCellGeneDataset(torch.utils.data.Dataset):

    def __init__(self, embed_split_file_name):
        asserts, _ = read_assets_from_h5(embed_split_file_name)
        self.embed_data = torch.tensor(asserts['embeddings'])
        self.gene_celltype_data = torch.tensor(asserts['y'])
        self.position_data = torch.tensor(asserts['pos'])
        self.edge_index = torch.tensor(asserts['edge_index'])
    
    def __len__(self):
        return len(self.embed_data)
    
    def __getitem__(self, idx):
        embed = self.embed_data[idx, :]
        gene_celltype = self.gene_celltype_data[idx, :]
        position = self.position_data[idx, :]

        return {'x': embed, 'y': gene_celltype, 'pos': position, 'edge_index': torch.tensor(0.)}



class ImgCellGeneDataset(torch.utils.data.Dataset):
    
    def __init__(self, split_file_name, data_root, img_transform=None, OTHER_DECONV_METHODS=None):
        """
        split_file_name: str
            The file name of the split file
        data_root: str
            The root directory of the data
        img_transform: torchvision.transforms
            The image transformation to apply to the images
        OTHER_DECONV_METHODS: str [default: None]
            The other deconvolution methods to use, such as 'Seurat', 'RCTD', 'tangram', etc.
            None for default Cell2location deconvolution methods
        """
        if OTHER_DECONV_METHODS is not None:
            sp_adata = sc.read_h5ad(os.path.join(data_root, 'sp.X_norm5e4_log1p_more_deconv.h5ad')) # read the spatial data

        train_slides = open(split_file_name).read().split('\n')
        self.batch_size = len(train_slides) # batch size is the number of slides

        batch_idx = []
        img_data = []
        gene_celltype_data = []
        position_data = []
        for idx, item in enumerate(train_slides):
            graph_data = torch.load(os.path.join(data_root, item+'.pt'))

            if OTHER_DECONV_METHODS is not None:
                sample_adata = sp_adata[sp_adata.obs['sample'] == item].copy() # get the sample data for the item slide
                print(sample_adata)
                cellabundance, celltype_name_list = self.get_deconv_cell_abundance(sample_adata, graph_data.pos, OTHER_DECONV_METHODS) # get the cell abundance data from the sample data
                print(f"Slide ID: {train_slides}, Using {OTHER_DECONV_METHODS} deconv. methods\n cellabundance cell types: {celltype_name_list}") 

                graph_data.y[:, 250:] = cellabundance # replace the cell abundance data in the graph data

            img_data.append(graph_data.x)
            gene_celltype_data.append(graph_data.y)
            position_data.append(graph_data.pos)
            batch_idx.extend([idx]*len(graph_data.x))

        self.img_data = torch.cat(img_data, dim=0)
        self.gene_celltype_data = torch.cat(gene_celltype_data, dim=0)
        self.position_data = torch.cat(position_data, dim=0)
        self.batch = torch.tensor(batch_idx).flatten()
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, idx):
        img = self.img_data[idx]
        gene_celltype = self.gene_celltype_data[idx]
        position = self.position_data[idx]

        if self.img_transform:
            img = self.img_transform(img)

        return {'x': img, 'y': gene_celltype, 'pos': position, 'edge_index': torch.tensor(0.)}

    ### revision 20250319, Get the cell abundance data from the h5ad file according to the position data from the PT file
    def get_deconv_cell_abundance(self, sample_adata, position_data, deconv_method='tangram'):
        """
        deconv_method: str
            The method used for deconvolution. Default is 'tangram', or RCTD'
        return:
            cellabundance: torch.tensor
                The cell abundance data from the h5ad file
            celltypes: list
                The cell type names for the cell abundance
        """
        # Compare the coordinates from position_data with spatial coordinates from sample_adata
        spatial_coords = sample_adata.obsm['spatial']
        pt_positions = position_data.numpy()  # Convert torch tensor to numpy array

        # Check if the number of spots matches between the two data sources
        assert len(pt_positions) == len(sample_adata), "Number of spots in PT file and AnnData do not match"

        # Create a KDTree from the spatial coordinates
        tree = KDTree(spatial_coords)

        # Find the closest spot for each PT position
        _, indices = tree.query(pt_positions)

        # Get the spot IDs corresponding to the matched indices
        # matched_spot_ids = [sample_adata.obs.index[i] for i in indices]

        # Check if the number of spots matches between the two data sources
        assert (sample_adata.obsm['spatial'][indices, :] - pt_positions).sum() == 0, "Spatial coordinates do not match"
        cellabundance = sample_adata.obsm[deconv_method].iloc[indices, :]

        return torch.tensor(cellabundance.values), list(cellabundance.columns)


def load_graph_pt_data(split_file_name, data_root):
    slides = open(split_file_name).read().split('\n')
    graph_list = list()
    for item in slides:
        graph_list.append(torch.load(os.path.join(data_root, item+'.pt')))
    dataset = Batch.from_data_list(graph_list)
    return dataset



class THItoGeneDataset(torch.utils.data.Dataset):
    
    def __init__(self, split_file_name, data_root, img_transform=None):

        train_slides = open(split_file_name).read().split('\n')
        self.batch_size = len(train_slides) # batch size is the number of slides

        batch_idx = []
        img_data = []
        gene_celltype_data = []
        position_data = []
        for idx, item in enumerate(train_slides):
            graph_data = torch.load(os.path.join(data_root, item+'.pt'))

            img_data.append(graph_data.x)
            gene_celltype_data.append(graph_data.y)
            position_data.append(graph_data.pos)
            batch_idx.extend([idx]*len(graph_data.x))

        self.train_slides = train_slides
        self.img_data = img_data
        self.gene_celltype_data = gene_celltype_data
        self.position_data = position_data
        self.batch = torch.tensor(batch_idx).flatten()
        
        if img_transform is None:
            self.img_transform = transforms.Compose([
                # transforms.ColorJitter(0.5, 0.5, 0.5), 
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees=180), 
                transforms.Resize((112, 112))])
        else:
            self.img_transform = img_transform

        spot_step = 110 if split_file_name.split("/")[-2] != "humanlung_cell2location" else 290
        self.position_data = [np.around(pos/spot_step).int() for pos in self.position_data] # convert pixel to position

        self.adj = [self.calcADJ(coord=pos, k=4, pruneTag='NA') for pos in self.position_data]

    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, idx): # each idx is a slide section
        img = self.img_data[idx]
        gene_celltype = self.gene_celltype_data[idx]
        position = self.position_data[idx]

        if self.img_transform:
            img = self.img_transform(img)

        return {'x': img, 'y': gene_celltype, 'pos': position, 'edge_index': torch.tensor(0.), 'adj': self.adj[idx], 'name': self.train_slides[idx]}

    @staticmethod
    def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA'):
        spatialMatrix = coord
        nodes = spatialMatrix.shape[0]
        Adj = torch.zeros((nodes, nodes))
        for i in np.arange(spatialMatrix.shape[0]):
            tmp = spatialMatrix[i, :].reshape(1, -1)
            distMat = distance.cdist(tmp, spatialMatrix, distanceType)
            if k == 0:
                k = spatialMatrix.shape[0] - 1
            res = distMat.argsort()[:k + 1]
            tmpdist = distMat[0, res[0][1:k + 1]]
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            for j in np.arange(1, k + 1):
                # No prune
                if pruneTag == 'NA':
                    Adj[i][res[0][j]] = 1.0
                elif pruneTag == 'STD':
                    if distMat[0, res[0][j]] <= boundary:
                        Adj[i][res[0][j]] = 1.0
                elif pruneTag == 'Grid':
                    if distMat[0, res[0][j]] <= 2.0:
                        Adj[i][res[0][j]] = 1.0
        return Adj

if __name__ == '__main__':
    pass