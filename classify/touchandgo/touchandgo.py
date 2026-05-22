''' material classification '''
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import xgboost as xgb  # pip install xgboost

# 1) Classification algorithms to try
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM (Linear Kernel)': SVC(kernel='linear'),
    'SVM (RBF Kernel)':    SVC(kernel='rbf'),
    'Random Forest':       RandomForestClassifier(n_estimators=100),
    'KNN (k=5)':           KNeighborsClassifier(n_neighbors=5),
    # 'Gradient Boosting':   GradientBoostingClassifier(),
    # 'Naive Bayes':         GaussianNB(),
    # 'XGBoost':             xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# 2) Data and transform settings
DATA_ROOT  = '/data/ssd/zhaoy/datasets/TouchandGoDataset-v2/dataset'
LABLE_FILE = '/data/ssd/zhaoy/datasets/TouchandGoDataset-v2/label.txt'
CATEG_FILE = '/data/ssd/zhaoy/datasets/TouchandGoDataset-v2/category_reference.txt'
BATCH_SIZE = 5000
transform = transforms.Compose([
    # transforms.Resize((64,64)),  # 降维到固定大小
    transforms.ToTensor()
])

# 3）Load Category Mapping and Samples
print(f'Load Category Mapping')
category_map = {}
with open(CATEG_FILE, 'r') as f:
    for line in f:
        if ':' not in line:
            continue
        name, idx = line.strip().split(':')
        idx = int(idx.strip().split()[0])
        category_map[idx] = name.strip()


print('Load Samples')
samples = []
with open(LABLE_FILE, 'r') as f:
    for line in f:
        path, cls = line.strip().split(',')
        cls = int(cls)
        if cls == -1:  # skip inconclusive
            continue
        full_path = os.path.join(DATA_ROOT, path.split("/")[0], "gelsight_frame", os.path.basename(path).replace('.jpg', '.png'))
        # full_path = os.path.join(DATA_ROOT, '_'.join(path.split('/')).replace('.jpg', '.png'))
        if os.path.isfile(full_path):
            samples.append((full_path, cls))
        
print(len(samples))
print(samples[0])


# 4) Train & evaluate each classifier
from PIL import Image
import numpy as np

X, y = [], []
SAMPLES_PER_CLASS = 80

import random
from collections import defaultdict

class_to_samples = defaultdict(list)
for img_path, cls in samples:
    class_to_samples[cls].append(img_path)

selected = []
for cls, img_paths in class_to_samples.items():
    if len(img_paths) < SAMPLES_PER_CLASS:
        print(f"Warning: class {cls} only has {len(img_paths)} samples (< {SAMPLES_PER_CLASS})")
        chosen = img_paths 
    else:
        chosen = random.sample(img_paths, SAMPLES_PER_CLASS)
    selected.extend([(p, cls) for p in chosen])

for img_path, cls in selected:
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)  # (3,64,64)
        feat = img_tensor.view(-1).numpy()  # flatten
        X.append(feat)
        y.append(cls)
    except Exception as e:
        print(f"Failed to load {img_path}: {e}")

X = np.array(X)
y = np.array(y)
print("Feature matrix:", X.shape)

print("Running PCA...")
pca = PCA(n_components=256, random_state=42)
X_reduced = pca.fit_transform(X)
print("Reduced feature matrix:", X_reduced.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

results = {}
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
