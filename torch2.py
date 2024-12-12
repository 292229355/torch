
import os
import cv2
import dlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader as GeoDataLoader
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

###############################################################################
#                               設定區域
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
assert os.path.exists(shape_predictor_path), "請提供正確的 shape_predictor_68_face_landmarks.dat 路徑"

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
superpoint_model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
superpoint_model.eval()
superpoint_model.to(device)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)


###############################################################################
#                               特徵提取相關函式
###############################################################################
def extract_face_keypoints(img_pth):
    img = cv2.imread(img_pth)
    if img is None:
        raise ValueError(f"Unable to load image at path: {img_pth}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None
    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)], dtype=np.int32)
    return gray, points

def extract_region(gray, points, point_indices):
    region_points = points[point_indices]
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, region_points, 255)
    region_img = cv2.bitwise_and(gray, gray, mask=mask)
    region_img_pil = Image.fromarray(region_img).convert("RGB")
    return region_img_pil

def extract_superpoint_features(img_pil, processor, superpoint_model, device, max_num_nodes=500, feature_dim=256):
    inputs = processor(img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = superpoint_model(**inputs)

    image_mask = outputs.mask[0]
    image_indices = torch.nonzero(image_mask).squeeze()
    if image_indices.numel() == 0:
        return None, None

    keypoints = outputs.keypoints[0][image_indices]
    descriptors = outputs.descriptors[0][image_indices]

    keypoint_coords = keypoints.cpu().numpy()
    descriptors = descriptors.cpu().numpy()

    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(1, -1)

    scaler = StandardScaler()
    descriptors = scaler.fit_transform(descriptors)

    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoint_coords = keypoint_coords[indices]

    return descriptors, keypoint_coords

def extract_face_region(img_pth, processor, superpoint_model, device, max_num_nodes=500, feature_dim=256):
    gray, points = extract_face_keypoints(img_pth)
    if gray is None:
        return None, None

    # 選擇若干臉部區域（可依需求微調）
    regions = {
        "left_eye": np.arange(42, 48),
        "right_eye": np.arange(36, 42),
        "nose": np.arange(27, 36),
        "face_contour": np.arange(0, 68)
    }

    all_des = []
    all_kp = []
    for _, r_idxs in regions.items():
        region_img_pil = extract_region(gray, points, r_idxs)
        des, kp = extract_superpoint_features(region_img_pil, processor, superpoint_model, device, max_num_nodes, feature_dim)
        if des is not None and kp is not None:
            all_des.append(des)
            all_kp.append(kp)

    if len(all_des) == 0:
        return None, None

    combined_descriptors = np.concatenate(all_des, axis=0)
    combined_keypoints = np.concatenate(all_kp, axis=0)

    return combined_descriptors, combined_keypoints


###############################################################################
#                               特徵篩選與圖建構
###############################################################################
def stable_sigmoid(x):
    return torch.where(
        x < 0, torch.exp(x) / (1 + torch.exp(x)), 1 / (1 + torch.exp(-x))
    )

def scm_pairwise(descriptors):
    dot_matrix = torch.matmul(descriptors, descriptors.t())
    norm_vec = torch.norm(descriptors, dim=1)
    cos_matrix = dot_matrix / (torch.ger(norm_vec, norm_vec) + 1e-8)
    x_matrix = cos_matrix / (norm_vec.sqrt().unsqueeze(1) + 1e-8)
    similarity_matrix = stable_sigmoid(x_matrix)
    return similarity_matrix

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
        scores = torch.matmul(Q, K.t()) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32).to(X.device))
        attention_scores = F.softmax(scores, dim=1)
        Matt = torch.sigmoid(torch.matmul(attention_scores, V))
        Matt = torch.matmul(Matt, torch.ones((self.dk, 1), device=X.device))
        Matt = Matt.squeeze(1)
        Matt = Matt.unsqueeze(1).repeat(1, X.size(0))
        return Matt

def filter_descriptors_adaptively(descriptors, device, method="similarity", retain_ratio=0.8):
    n = descriptors.size(0)
    if n == 0:
        return descriptors, torch.arange(n)

    if method == "similarity":
        similarity_matrix = scm_pairwise(descriptors)
        avg_sim = similarity_matrix.mean(dim=1)
        score = avg_sim
    else:
        desc_norm = torch.norm(descriptors, dim=1)
        score = desc_norm

    threshold_index = int(n * retain_ratio)
    _, sorted_indices = torch.sort(score, descending=True)
    selected_indices = sorted_indices[:threshold_index]
    filtered_descriptors = descriptors[selected_indices]

    return filtered_descriptors, selected_indices

def add_positional_features(x, pos):
    # 將座標標準化後與特徵串接
    pos_norm = (pos - pos.mean(dim=0)) / (pos.std(dim=0) + 1e-6)
    x_with_pos = torch.cat([x, pos_norm], dim=1)
    return x_with_pos

def extract_descriptors_and_build_graph(
    img_pth,
    processor,
    superpoint_model,
    device,
    max_num_nodes=500,
    feature_dim=256,
    eta=0.5,
    attention_module=None,
    threshold=0.5,
    adaptive_filter_method="similarity",
    adaptive_retain_ratio=0.8
):
    combined_descriptors, combined_keypoints = extract_face_region(
        img_pth, processor, superpoint_model, device, max_num_nodes, feature_dim
    )

    if combined_descriptors is None or combined_keypoints is None:
        x = torch.zeros((1, feature_dim), dtype=torch.float).to(device)
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        pos = torch.zeros((1, 2), dtype=torch.float).to(device)
        edge_attr = torch.empty((0,), dtype=torch.float).to(device)
        return x, edge_index, pos, edge_attr, torch.zeros((1,1), dtype=torch.float).to(device)

    x = torch.from_numpy(combined_descriptors).float().to(device)
    pos = torch.from_numpy(combined_keypoints).float().to(device)

    # 篩選特徵
    x_filtered, selected_indices = filter_descriptors_adaptively(
        x, device, method=adaptive_filter_method, retain_ratio=adaptive_retain_ratio
    )
    pos_filtered = pos[selected_indices]

    n = x_filtered.size(0)
    if n < 2:
        adjacency_matrix = torch.zeros((n, n), dtype=torch.float).to(device)
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        edge_attr = torch.empty((0,), dtype=torch.float).to(device)
        return x_filtered, edge_index, pos_filtered, edge_attr, adjacency_matrix

    # 加入位置特徵
    x_filtered = add_positional_features(x_filtered, pos_filtered)
    # x_filtered.shape = (num_nodes, feature_dim + 2)

    similarity_matrix = scm_pairwise(x_filtered)
    if attention_module is not None:
        attention_matrix = attention_module(x_filtered)
    else:
        attention_matrix = similarity_matrix

    adjacency_matrix = eta * similarity_matrix + (1 - eta) * attention_matrix
    mask = adjacency_matrix >= threshold
    edge_indices = mask.nonzero(as_tuple=False).t()
    edge_weights = adjacency_matrix[edge_indices[0], edge_indices[1]]

    return x_filtered, edge_indices, pos_filtered, edge_weights, adjacency_matrix


###############################################################################
#                               Dataset 定義
###############################################################################
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
        threshold=0.5
    ):
        super(DescriptorGraphDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.max_num_nodes = max_num_nodes
        self.feature_dim = feature_dim
        self.eta = eta
        self.dk = dk
        self.attention_module = attention_module
        self.threshold = threshold

        self.files = sorted(
            [
                os.path.join(path, x)
                for x in os.listdir(path)
                if x.lower().endswith(".jpg") or x.lower().endswith(".png")
            ]
        )

        if len(self.files) == 0:
            raise FileNotFoundError(f"No image files found in {path}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        data = extract_descriptors_and_build_graph(
            fname,
            processor,
            superpoint_model,
            device,
            max_num_nodes=self.max_num_nodes,
            feature_dim=self.feature_dim,
            eta=self.eta,
            attention_module=self.attention_module,
            threshold=self.threshold
        )

        x, edge_index, pos, edge_attr, adjacency_matrix = data

        # 從檔名或資料夾結構推測label, 假設檔名為 xxx_real_1.jpg 或 xxx_fake_0.jpg
        # 請依自身實務需求修改此規則
        try:
            # 假設檔名格式: [prefix]_[label].jpg
            file_parts = os.path.basename(fname).split("_")
            label_str = file_parts[1].split(".")[0]
            label = int(label_str)
        except:
            label = 0

        graph_data = Data(
            x=x.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            y=torch.tensor([label], dtype=torch.float, device=device),
            pos=pos.to(device),
        )

        return graph_data


###############################################################################
#                               模型定義
###############################################################################
class EnhancedGATClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedGATClassifier, self).__init__()
        # 使用GATv2Conv
        self.conv1 = GATv2Conv(input_dim, 128, heads=4, concat=True, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(128 * 4)
        self.conv2 = GATv2Conv(128 * 4, 64, heads=4, concat=False, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is not None and edge_attr.size(0) == 0:
            edge_attr = None
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
        return x

###############################################################################
#                               訓練/驗證函數
###############################################################################
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr).squeeze()
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
            logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr).squeeze()
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


###############################################################################
#                                 主程式
###############################################################################
if __name__ == "__main__":
    # 您的資料集路徑，結構假設: dataset_dir/train, dataset_dir/valid, dataset_dir/test
    dataset_dir = "Inpaint_dataset"
    exp_name = "Inpaint_Enhanced"

    # 初始化注意力模塊
    attention_module = AttentionModule(input_dim=256+2, dk=64).to(device) 
    attention_module.eval()

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

    batch_size = 128
    train_loader = GeoDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = GeoDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # 從train_set取樣本確定input_dim
    sample_data = None
    for data in train_set:
        if data is not None:
            sample_data = data
            break
    if sample_data is None:
        print("No valid data found in training set.")
        exit()

    input_dim = sample_data.x.size(1) # 已加入pos後之維度

    model = EnhancedGATClassifier(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_epochs = 25
    patience = 5
    best_acc = 0
    stale = 0

    for epoch in range(n_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_labels, valid_probs = validate(model, valid_loader, criterion)

        print(
            f"[Epoch {epoch + 1:03d}/{n_epochs:03d}] "
            f"Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f} | "
            f"Valid Loss: {valid_loss:.5f}, Valid Acc: {valid_acc:.5f}"
        )

        scheduler.step()

        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{exp_name}_best.ckpt")
            best_acc = valid_acc
            stale = 0
            best_valid_labels = valid_labels
            best_valid_probs = valid_probs
        else:
            stale += 1
            if stale > patience:
                print(f"No improvement in {patience} consecutive epochs, early stopping")
                break

    # 測試集推論
    test_set = DescriptorGraphDataset(
        os.path.join(dataset_dir, "test"),
        mode="test",
        attention_module=attention_module,
    )
    test_loader = GeoDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model_best = EnhancedGATClassifier(input_dim).to(device)
    model_best.load_state_dict(torch.load(f"models/{exp_name}_best.ckpt", map_location=device))
    model_best.eval()

    prediction = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            logits = model_best(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr).squeeze()
            preds = torch.sigmoid(logits)
            prediction.extend((preds > 0.5).long().cpu().numpy())

    import pandas as pd
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
        plt.plot(fpr, tpr, color="darkorange", lw=lw, label=f"ROC curve (AUC = {roc_auc:.4f})")
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
