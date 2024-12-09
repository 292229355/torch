# preprocessing.py

import os
import torch
import numpy as np
import cv2
import dlib
from PIL import Image
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
# Set environment variable for TensorFlow (if needed elsewhere)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize processors and models
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
superpoint_model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
superpoint_model.eval()
superpoint_model.to(device)

# Initialize Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is in the working directory or provide the full path

def extract_descriptors_and_build_graph2(img_pth, max_num_nodes=500, feature_dim=256, k=8):
    img = cv2.imread(img_pth)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        print(f"No faces detected in image {img_pth}.")
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)], dtype=np.int32)

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, points, 255)
    face_img = cv2.bitwise_and(gray, gray, mask=mask)

    face_img_pil = Image.fromarray(face_img).convert("RGB")

    inputs = processor(face_img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = superpoint_model(**inputs)

    image_mask = outputs.mask[0]
    image_indices = torch.nonzero(image_mask).squeeze()
    if image_indices.numel() == 0:
        print(f"No keypoints detected in image {img_pth}.")
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    keypoints = outputs.keypoints[0][image_indices]
    descriptors = outputs.descriptors[0][image_indices]

    keypoint_coords = keypoints.cpu().numpy()
    descriptors = descriptors.cpu().numpy()

    scaler = StandardScaler()
    descriptors = scaler.fit_transform(descriptors)

    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoint_coords = keypoint_coords[indices]

    num_nodes = descriptors.shape[0]

    dists = np.linalg.norm(
        keypoint_coords[:, np.newaxis, :] - keypoint_coords[np.newaxis, :, :],
        axis=2
    )
    np.fill_diagonal(dists, np.inf)

    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        neighbor_indices = np.argsort(dists[i])[:k]
        for j in neighbor_indices:
            distance = dists[i, j]
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append([1 / (distance + 1e-5)])
            edge_attr.append([1 / (distance + 1e-5)])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.from_numpy(descriptors).float().to(device)
    y = torch.from_numpy(keypoint_coords).float().to(device)

    return x, edge_index, y, edge_attr

def stable_sigmoid(x):
    return torch.where(
        x < 0, torch.exp(x) / (1 + torch.exp(x)), 1 / (1 + torch.exp(-x))
    )

def scm(des1, des2):
    dotproduct = torch.sum(des1 * des2, dim=1) / (
        torch.norm(des1, dim=1) * torch.norm(des2, dim=1) + 1e-8
    )
    x = dotproduct / (torch.norm(des1, dim=1) ** 0.5 + 1e-8)
    similarity = stable_sigmoid(x)
    return similarity

def cosine_similarity(des1, des2):
    """
    Compute the cosine similarity between two tensors.
    
    Args:
        des1 (torch.Tensor): Tensor of shape (N, D) where N is the number of samples and D is the dimensionality.
        des2 (torch.Tensor): Tensor of shape (N, D) where N is the number of samples and D is the dimensionality.
    
    Returns:
        torch.Tensor: Cosine similarity for each pair in the batch, shape (N,).
    """
    return F.cosine_similarity(des1, des2, dim=1)

def andm(A, gamma, beta):
    n = A.size(0)
    mean_A = torch.mean(A, dim=1, keepdim=True)
    Ti = gamma * mean_A + beta
    AT = torch.where(A > Ti, A, torch.zeros_like(A))
    AN = F.softmax(AT, dim=1) * (AT > 0).float()
    return AN

class AttentionModule(nn.Module):
    def __init__(self, input_dim, dk=64):
        super(AttentionModule, self).__init__()
        self.Q = nn.Linear(input_dim, dk, bias=False)
        self.K = nn.Linear(input_dim, dk, bias=False)
        self.V = nn.Linear(input_dim, dk, bias=False)
        self.dk = dk

    def forward(self, X):
        Q = self.Q(X)
        K = self.K(X)
        V = self.V(X)

        scores = torch.matmul(Q, K.t()) / torch.sqrt(
            torch.tensor(self.dk, dtype=torch.float32).to(X.device)
        )
        attention_scores = F.softmax(scores, dim=1)

        Matt = torch.sigmoid(torch.matmul(attention_scores, V))
        Matt = torch.matmul(Matt, torch.ones((self.dk, 1), device=X.device))
        Matt = Matt.squeeze(1)
        Matt = Matt.unsqueeze(1).repeat(1, X.size(0))

        return Matt

def calculate_adjacency_matrix_with_andm(X, eta=0.5, dk=64, attention_module=None, similarity_metric="scm", zeta=0.3):
    """
    Computes an adjacency matrix combining SCM scores, cosine similarity, and an additional hybrid metric.
    Args:
        X (torch.Tensor): Input feature matrix of shape (N, D).
        eta (float): Weighting parameter for SCM or cosine similarity scores.
        dk (int): Dimension of the key and query vectors in the attention module.
        attention_module (AttentionModule, optional): Predefined attention module.
        similarity_metric (str): Choice of similarity metric ("scm" or "cosine").
        zeta (float): Weighting parameter for the hybrid additional metric.
    Returns:
        torch.Tensor: Final adjacency matrix of shape (N, N).
    """
    n, d = X.size()

    with torch.no_grad():
        # Similarity calculation based on selected metric
        if similarity_metric == "cosine":
            # Compute cosine similarity
            norm_X = F.normalize(X, p=2, dim=1)
            similarity_scores = torch.matmul(norm_X, norm_X.t())
        elif similarity_metric == "scm":
            # Compute SCM scores
            similarity_scores = torch.zeros((n, n), device=X.device)
            for i in range(n):
                for j in range(n):
                    similarity_scores[i, j] = scm(X[i].unsqueeze(0), X[j].unsqueeze(0))
        else:
            raise ValueError("Invalid similarity_metric. Choose 'scm' or 'cosine'.")

        similarity_scores = torch.clamp(similarity_scores, min=-1.0, max=1.0)

        # Additional hybrid metric: Euclidean distance converted to similarity
        dist_matrix = torch.cdist(X, X, p=2)  # Compute pairwise Euclidean distances
        hybrid_scores = torch.exp(-dist_matrix)  # Convert to similarity using Gaussian kernel

    if attention_module is not None:
        Matt = attention_module(X)
    else:
        Matt = torch.zeros(n, n).to(X.device)

    # Combine similarity scores, hybrid scores, and attention scores
    A = eta * similarity_scores + (1 - eta - zeta) * Matt + zeta * hybrid_scores

    gamma = torch.rand(n, 1, device=X.device)
    beta = torch.rand(n, 1, device=X.device)

    AN = andm(A, gamma, beta)

    return AN



def extract_descriptors_and_build_graph_with_andm(img_pth, max_num_nodes=500, feature_dim=256, eta=0.5, dk=64, attention_module=None):
    x, edge_index, y, edge_attr = extract_descriptors_and_build_graph2(
        img_pth, max_num_nodes=max_num_nodes, feature_dim=feature_dim
    )

    if x.size(0) == 0:
        print("No nodes detected, returning zero adjacency matrix.")
        return (
            x,
            edge_index,
            y,
            edge_attr,
            torch.zeros((1, 1), dtype=torch.float).to(device),
        )

    feature_matrix = x

    adjacency_matrix = calculate_adjacency_matrix_with_andm(
        feature_matrix, eta=eta, dk=dk, attention_module=attention_module
    )

    return x, edge_index, y, edge_attr, adjacency_matrix

class DescriptorGraphDataset(Dataset):
    def __init__(self, path, mode="train", max_num_nodes=500, feature_dim=256, eta=0.5, dk=64, attention_module=None):
        super(DescriptorGraphDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.max_num_nodes = max_num_nodes
        self.feature_dim = feature_dim
        self.eta = eta
        self.dk = dk
        self.attention_module = attention_module

        self.files = sorted([
            os.path.join(path, x)
            for x in os.listdir(path)
            if x.endswith(".jpg") or x.endswith(".png")
        ])

        if len(self.files) == 0:
            raise FileNotFoundError(f"No image files found in {path}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        data = extract_descriptors_and_build_graph_with_andm(
            fname,
            max_num_nodes=self.max_num_nodes,
            feature_dim=self.feature_dim,
            eta=self.eta,
            dk=self.dk,
            attention_module=self.attention_module,
        )

        x, edge_index, pos, edge_attr, adjacency_matrix = data

        try:
            file_parts = os.path.basename(fname).split("_")
            label_str = file_parts[1].split(".")[0]
            label = int(label_str)
        except:
            label = -1  # Use -1 for missing or invalid labels

        if adjacency_matrix.size(0) > 1:
            adj_indices = torch.nonzero(adjacency_matrix, as_tuple=False).t()
            adj_weights = adjacency_matrix[adj_indices[0], adj_indices[1]]
        else:
            adj_indices = torch.empty((2, 0), dtype=torch.long).to(device)
            adj_weights = torch.empty((0,), dtype=torch.float).to(device)

        graph_data = Data(
            x=x.to(device),
            edge_index=adj_indices,
            edge_attr=adj_weights,
            y=torch.tensor([label], dtype=torch.float),
            pos=pos,
        )

        return graph_data
