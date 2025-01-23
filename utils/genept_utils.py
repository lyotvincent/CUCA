import os

import pickle
import pandas as pd
import numpy as np


"""
load_genept_data: Load GenePT embeddings for genes in the dataset
gene_embedding from https://github.com/yiqunchen/GenePT
Args:
    genes: list of genes in the dataset
    genept_path: root_path to the GenePT embeddings (default=None)
    file_name: name of the GenePT embeddings file (default="GenePT_gene_embedding_ada_text.pickle")
    embed_dim: dimension of the GenePT embeddings (default=1536)
"""
def load_genept_data(genes, genept_path=None, 
                     file_name = "GenePT_gene_embedding_ada_text.pickle", 
                     embed_dim=1536):
    assert genept_path is not None, "Please provide the path to the GenePT embeddings"

    with open(os.path.join(genept_path, file_name), "rb") as f:
        genept_embedding = pickle.load(f)

    dataset_embeds = {}
    for single_gene in genes:
        if single_gene not in genept_embedding.keys():
            print(f"{single_gene} not in GenePT embedding, filled with zeros")
            dataset_embeds[single_gene] = np.array([0]*embed_dim)
        else:
            dataset_embeds[single_gene] = np.array(genept_embedding[single_gene])

    dataset_embeds = pd.DataFrame(dataset_embeds).T
    return dataset_embeds.values
