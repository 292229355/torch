import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import cv2
import dlib
from PIL import Image
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 設置環境變量
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化處理器和模型
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
superpoint_model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
superpoint_model.eval().to(device)

# Dlib 初始化
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 初始化標準化器
scaler = StandardScaler()

def extract_descriptors_and_build_graph(
    img_pth,
    max_num_nodes=500,
    feature_dim=256,
    k=5
):
    img = cv2.imread(img_pth)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        print(f"No faces detected in image {img_pth}.")
        return torch.zeros((1, feature_dim), dtype=torch.float).to(device), \
               torch.empty((2, 0), dtype=torch.long).to(device), \
               torch.zeros((1, 2), dtype=torch.float).to(device), \
               torch.empty((0, 1), dtype=torch.float).to(device)

    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)], dtype=np.int32)

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, points, 255)
    face_img = cv2.bitwise_and(gray, gray, mask=mask)

    face_img_pil = Image.fromarray(face_img).convert("RGB")

    # 使用 SuperPoint 進行特徵提取
    inputs = processor(face_img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = superpoint_model(**inputs)

    image_mask = outputs.mask[0]
    image_indices = torch.nonzero(image_mask).squeeze()
    if image_indices.numel() == 0:
        print(f"No keypoints detected in image {img_pth}.")
        return torch.zeros((1, feature_dim), dtype=torch.float).to(device), \
               torch.empty((2, 0), dtype=torch.long).to(device), \
               torch.zeros((1, 2), dtype=torch.float).to(device), \
               torch.empty((0, 1), dtype=torch.float).to(device)

    keypoints = outputs.keypoints[0][image_indices].cpu().numpy()
    descriptors = outputs.descriptors[0][image_indices].cpu().numpy()

    # 標準化描述子
    descriptors = scaler.fit_transform(descriptors)

    # 限制節點數量
    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoint_coords = keypoints[indices]
    else:
        keypoint_coords = keypoints

    num_nodes = descriptors.shape[0]

    # 使用 KNN 構建圖
    dists = np.linalg.norm(keypoint_coords[:, np.newaxis, :] - keypoint_coords[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)

    # 找到每個節點的 K 個最近鄰
    knn_indices = np.argsort(dists, axis=1)[:, :k].reshape(-1)
    source_nodes = np.repeat(np.arange(num_nodes), k)
    target_nodes = knn_indices
    distances = dists[source_nodes, target_nodes]

    # 添加雙向邊
    edge_index = np.vstack((np.concatenate([source_nodes, target_nodes]),
                            np.concatenate([target_nodes, source_nodes])))
    edge_attr = np.concatenate([1 / (distances + 1e-5), 1 / (distances + 1e-5)])

    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1).to(device)

    x = torch.from_numpy(descriptors).float().to(device)
    y = torch.from_numpy(keypoint_coords).float().to(device)

    return x, edge_index, y, edge_attr

def stable_sigmoid(x):
    return torch.where(
        x < 0, torch.exp(x) / (1 + torch.exp(x)), 1 / (1 + torch.exp(-x))
    )

class AttentionModule(nn.Module):
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

        scores = torch.matmul(Q, K.t()) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32).to(X.device))
        attention_scores = F.softmax(scores, dim=1)  # (n x n)

        Matt = torch.sigmoid(torch.matmul(attention_scores, V))  # (n x dk)
        Matt = torch.matmul(Matt, torch.ones((self.dk, 1), device=X.device))  # (n x 1)
        Matt = Matt.squeeze(1)  # (n,)

        Matt = Matt.unsqueeze(1).repeat(1, X.size(0))  # (n x n)

        return Matt

def andm(A, gamma, beta):
    mean_A = torch.mean(A, dim=1, keepdim=True)  # (n x 1)
    Ti = gamma * mean_A + beta  # (n x 1)
    AT = torch.where(A > Ti, A, torch.zeros_like(A))  # (n x n)
    AN = F.softmax(AT, dim=1) * (AT > 0).float()  # (n x n)
    return AN

def calculate_adjacency_matrix_with_andm(X, eta=0.5, dk=64, attention_module=None):
    n, d = X.size()

    # SCM: 計算相似度矩陣
    norm_X = F.normalize(X, p=2, dim=1)  # (n x d)
    Msim = torch.matmul(norm_X, norm_X.t())  # (n x n)
    Msim = torch.clamp(Msim, min=-1.0, max=1.0)

    # SCM scaling
    dotproduct = torch.sum(X.unsqueeze(1) * X.unsqueeze(0), dim=2) / (
        torch.norm(X, dim=1).unsqueeze(1) * torch.norm(X, dim=1).unsqueeze(0) + 1e-8
    )  # (n x n)
    x = dotproduct / (torch.norm(X, dim=1).unsqueeze(1) ** 0.5 + 1e-8)  # (n x n)
    scm_scores = stable_sigmoid(x)  # (n x n)

    # SAM: 自注意力機制
    if attention_module is not None:
        Matt = attention_module(X)  # (n x n)
    else:
        Matt = torch.zeros(n, n).to(X.device)

    # 結合 SCM 和 SAM
    A = eta * scm_scores + (1 - eta) * Matt  # (n x n)

    # 定義 ANDM 的可訓練參數 gamma 和 beta
    gamma = torch.rand(n, 1, device=X.device, requires_grad=True)  # (n x 1)
    beta = torch.rand(n, 1, device=X.device, requires_grad=True)  # (n x 1)

    # 應用 ANDM
    AN = andm(A, gamma, beta)  # (n x n)

    return AN

def extract_descriptors_and_build_graph_with_andm(
    img_pth, max_num_nodes=500, feature_dim=256, eta=0.5, dk=64, attention_module=None
):
    x, edge_index, y, edge_attr = extract_descriptors_and_build_graph(
        img_pth, max_num_nodes=max_num_nodes, feature_dim=feature_dim
    )

    if x.size(0) == 0:
        print("No nodes detected, returning zero adjacency matrix.")
        adjacency_matrix = torch.zeros((1, 1), dtype=torch.float).to(device)
    else:
        adjacency_matrix = calculate_adjacency_matrix_with_andm(
            x, eta=eta, dk=dk, attention_module=attention_module
        )

    return x, edge_index, y, edge_attr, adjacency_matrix



class DescriptorGraphDataset(Dataset):
    def __init__(
        self,
        path,
        mode="train",
        max_num_nodes=500,
        feature_dim=256,
        eta=0.5,
        dk=64,
        attention_module=None,
    ):
        """
        Dataset for extracting graph-based descriptors from images.

        Args:
            path (str): Path to the directory containing image files.
            mode (str): Dataset mode ('train', 'val', or 'test').
            max_num_nodes (int): Maximum number of nodes in the graph.
            feature_dim (int): Dimensionality of the feature descriptors.
            eta (float): Balance parameter for combining SCM and SAM.
            dk (int): Dimensionality of attention keys in SAM.
            attention_module (nn.Module): Instance of AttentionModule.
        """
        super(DescriptorGraphDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.max_num_nodes = max_num_nodes
        self.feature_dim = feature_dim
        self.eta = eta
        self.dk = dk
        self.attention_module = attention_module

        # Get list of image files
        self.files = sorted(
            [
                os.path.join(path, x)
                for x in os.listdir(path)
                if x.endswith(".jpg") or x.endswith(".png")
            ]
        )

        if len(self.files) == 0:
            raise FileNotFoundError(f"No image files found in {path}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Extract graph data and adjacency matrix
        data = extract_descriptors_and_build_graph_with_andm(
            fname,
            max_num_nodes=self.max_num_nodes,
            feature_dim=self.feature_dim,
            eta=self.eta,
            dk=self.dk,
            attention_module=self.attention_module,
        )

        # Unpack the data
        x, edge_index, pos, edge_attr, adjacency_matrix = data

        # Determine the label from the filename
        try:
            file_parts = os.path.basename(fname).split("_")
            label_str = file_parts[1].split(".")[0]
            label = int(label_str)
        except:
            label = -1  # Use -1 for missing or invalid labels

        # Convert adjacency matrix to PyTorch Geometric edge index and edge attributes
        if adjacency_matrix.size(0) > 1:  # Ensure adjacency matrix is not empty
            adj_indices = torch.nonzero(adjacency_matrix, as_tuple=False).t()
            adj_weights = adjacency_matrix[adj_indices[0], adj_indices[1]]
        else:
            adj_indices = torch.empty((2, 0), dtype=torch.long).to(device)
            adj_weights = torch.empty((0,), dtype=torch.float).to(device)

        # Create PyTorch Geometric data object
        graph_data = Data(
            x=x.to(device),
            edge_index=adj_indices,
            edge_attr=adj_weights,
            y=torch.tensor([label], dtype=torch.float),
            pos=pos,
        )

        return graph_data


class GATClassifier(nn.Module):
    def __init__(self, input_dim):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(input_dim, 128, heads=4, concat=True, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(128 * 4)
        self.conv2 = GATConv(128 * 4, 256, heads=4, concat=True, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(256 * 4)
        self.conv3 = GATConv(256 * 4, 64, heads=4, concat=False, edge_dim=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is not None and edge_attr.size(0) == 0:
            edge_attr = None  # 如果沒有邊特徵，設置為 None
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # 返回 logits



def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
        ).squeeze()
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        preds = torch.sigmoid(logits)
        acc = ((preds > 0.5).float() == batch.y).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
    return total_loss / len(loader), total_acc / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = batch.to(device)
            logits = model(
                batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
            ).squeeze()
            loss = criterion(logits, batch.y)
            preds = torch.sigmoid(logits)
            acc = ((preds > 0.5).float() == batch.y).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(preds.cpu().numpy())
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc, all_labels, all_probs


if __name__ == "__main__":
    dataset_dir = "dataset\ADM_dataset"
    exp_name = "ADM"

    os.makedirs("models", exist_ok=True)

    # 初始化注意力模塊
    attention_module = AttentionModule(input_dim=256, dk=64).to(device)
    attention_module.eval()  # 如果不需要訓練 SAM，可以設置為 eval()

    train_set = DescriptorGraphDataset(
        os.path.join(dataset_dir, "train"),
        mode="train",
        attention_module=attention_module,
    )
    valid_set = DescriptorGraphDataset(
        os.path.join(dataset_dir, "valid"),
        mode="valid",
        attention_module=attention_module,
    )

    # DataLoader
    batch_size = 64  # 減少批次大小
    train_loader = GeoDataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    valid_loader = GeoDataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    sample_data = None
    for data in train_set:
        if data is not None:
            sample_data = data
            break
    if sample_data is None:
        print("No valid data found in training set.")
        exit()
    input_dim = sample_data.num_node_features

    model = GATClassifier(input_dim).to(device)
    # Binary classification 使用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_epochs = 50  # 增加 epoch 數量以更好地訓練模型
    patience = 10
    best_acc = 0
    stale = 0
    train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_labels, valid_probs = validate(
            model, valid_loader, criterion
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        print(
            f"[Epoch {epoch + 1:03d}/{n_epochs:03d}] "
            f"Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f} | "
            f"Valid Loss: {valid_loss:.5f}, Valid Acc: {valid_acc:.5f}"
        )

        scheduler.step()

        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            torch.save(model.state_dict(), f"models/{exp_name}_best.ckpt")
            best_acc = valid_acc
            stale = 0
            # 用來繪製 ROC
            best_valid_labels = valid_labels
            best_valid_probs = valid_probs
        else:
            stale += 1
            if stale > patience:
                print(
                    f"No improvement in {patience} consecutive epochs, early stopping"
                )
                break

    test_set = DescriptorGraphDataset(
        os.path.join(dataset_dir, "test"),
        mode="test",
        attention_module=attention_module,
    )
    test_loader = GeoDataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model_file = f"models/{exp_name}_best.ckpt"
    model_best = GATClassifier(input_dim).to(device)
    model_best.load_state_dict(
        torch.load(model_file, map_location=device)
    )  # 使用 map_location
    model_best.eval()

    prediction = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            logits = model_best(
                batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
            ).squeeze()
            preds = torch.sigmoid(logits)
            prediction.extend((preds > 0.5).long().cpu().numpy())

    df = pd.DataFrame()
    df["Id"] = [str(i).zfill(4) for i in range(1, len(test_set) + 1)]
    df["Category"] = prediction
    df.to_csv("submission.csv", index=False)
    print("Prediction results saved to submission.csv")

    if "best_valid_labels" in locals() and "best_valid_probs" in locals():
        best_valid_labels = np.array(best_valid_labels)
        best_valid_probs = np.array(best_valid_probs)

        fpr, tpr, thresholds = roc_curve(best_valid_labels, best_valid_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (AUC = %0.4f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.show()
    else:
        print("No validation data available to compute ROC curve.")