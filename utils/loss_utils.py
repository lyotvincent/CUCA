import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean'):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true))
        return loss


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        symmetric:   If True, the loss is calculated symmetrically. (default: True)
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.01, symmetric=False, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.symmetric = symmetric

    def forward(self, query, positive_key, negative_keys=None):
        return self.info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode,
                        symmetric=self.symmetric)


    def info_nce(self, query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired', symmetric=False):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors (normalize by euclidean distance)
        query, positive_key, negative_keys = self.normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ self.transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ self.transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            logits = query @ self.transpose(positive_key)

            # Positive keys are the entries on the diagonal (class indices for each row)
            labels = torch.arange(len(query), device=query.device)
            
            # symmetric contrastive loss 
            if symmetric:
                logits2 = positive_key @ self.transpose(query)
                loss = 0.5*F.cross_entropy(logits / temperature, labels, reduction=reduction) + 0.5*F.cross_entropy(logits2 / temperature, labels, reduction=reduction)
            else:
                loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)
                
        return loss

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]
    


class PearsonLoss(nn.Module):
    def __init__(self, mean_dim=0, reduction="mean"):
        """
        Pearson Correlation Coefficient Loss function.
        """
        super(PearsonLoss, self).__init__()
        self.mean_dim = mean_dim
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        r"""
        Pearson Correlation Coefficient Loss function.

        args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): True values
        """
        y_true = y_true - torch.mean(y_true, self.mean_dim)
        y_pred = y_pred - torch.mean(y_pred, self.mean_dim)
        y_true = y_true / torch.sqrt(torch.sum(y_true**2, self.mean_dim) + 1e-8)
        y_pred = y_pred / torch.sqrt(torch.sum(y_pred**2, self.mean_dim) + 1e-8)

        pearson_corr = torch.sum(y_true * y_pred, self.mean_dim)

        if self.reduction == "mean":
            pearson_loss = (1 - pearson_corr).mean()
        elif self.reduction == "sum":
            pearson_loss = (1 - pearson_corr).sum()
        # pearson_loss = 1 - pearsonr(y_true, y_pred)[0]

        return pearson_loss
