''' material classification (JSON version) '''
import os
import json
import warnings
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

# 1) Classification algorithms
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

DATASET = 'ObjectFolder_1.0'   
DATA_ROOT  = f'/data/ssd/zhaoy/datasets/{DATASET}/touch/image'
LABEL_JSON = f'/data/ssd/zhaoy/datasets/{DATASET}/label.json'
BATCH_SIZE = 5000
transform = transforms.Compose([
    transforms.Resize((64,64)),  # 固定大小
    transforms.ToTensor()
])

print("Load Labels from JSON")
with open(LABEL_JSON, 'r') as f:
    folder_to_cls = json.load(f)


print("Collecting Samples")
samples = []
for folder, cls in folder_to_cls.items():
    folder_path = os.path.join(DATA_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue
    for fname in os.listdir(folder_path):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, fname)
            samples.append((full_path, cls))

print(f"Total samples: {len(samples)}")
print(samples[0])


X, y = [], []
SAMPLES_PER_CLASS = 5000 

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


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

import numpy as np

print("Train distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

print("\nTest distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")


from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import numpy as np
from PIL import Image

results = {}
for name, clf in classifiers.items():
    print(f"\n===== Training {name} =====")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 1) 基本指标
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    results[name] = {"accuracy": acc, "balanced_acc": bal_acc}

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Balanced Accuracy: {bal_acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))