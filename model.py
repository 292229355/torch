# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class AttentionModule(nn.Module):
    """
    Self-Attention Mechanism (SAM) implemented in PyTorch.
    """

    def __init__(self, input_dim, dk=64):
        super(AttentionModule, self).__init__()
        self.Q = nn.Linear(input_dim, dk, bias=False)
        self.K = nn.Linear(input_dim, dk, bias=False)
        self.V = nn.Linear(input_dim, dk, bias=False)
        self.dk = dk

    def forward(self, X):
        Q = self.Q(X)  # (n x dk)
        K = self.K(X)  # (n x dk)
        V = self.V(X)  # (n x dk)

        scores = torch.matmul(Q, K.t()) / torch.sqrt(
            torch.tensor(self.dk, dtype=torch.float32).to(X.device)
        )  # (n x n)
        attention_scores = F.softmax(scores, dim=1)  # (n x n)

        Matt = torch.sigmoid(torch.matmul(attention_scores, V))  # (n x dk)
        Matt = torch.matmul(Matt, torch.ones((self.dk, 1), device=X.device))  # (n x 1)
        Matt = Matt.squeeze(1)  # (n,)

        Matt = Matt.unsqueeze(1).repeat(1, X.size(0))  # (n x n)

        return Matt

class GATClassifier(nn.Module):
    def __init__(self, input_dim):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(input_dim, 128, heads=4, concat=True, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(128 * 4)
        self.conv2 = GATConv(128 * 4, 64, heads=4, concat=False, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is not None and edge_attr.size(0) == 0:
            edge_attr = None  # If no edge features, set to None
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x  # Returns logits
