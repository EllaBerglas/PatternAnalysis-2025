"""
showing example usage of your trained model. Print out any results and / or provide
visualisations where applicable
"""
from collections import defaultdict
import os
from dataset import test_loader
import torch # type: ignore
from parameters import MODEL_FILENAME
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
from modules import ConvNeXt
import matplotlib.pyplot as plt # type: ignore
from parameters import MODEL_CONFIG, COMPILE
from sklearn.metrics import confusion_matrix, classification_report  # type: ignore

THRESHOLD = 0.70

def eval_accuracy(model, device):
    all_labels = []
    all_preds = []
    all_probs = []
    model.eval()
    print(f"Loaded model")
    class_names = ["AD", "NC"]

    with torch.no_grad():
        for (volumes, labels) in tqdm(test_loader, desc="Test Set"):
            volumes = volumes.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            output = model(volumes)
            probs = torch.sigmoid(output)
            all_probs.extend(probs.cpu().numpy())
            predicted = (probs > THRESHOLD).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    test_accuracy = np.mean(all_labels == all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # plot accuracies over different thresholds
    all_probs = np.array(all_probs)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.85,  0.9, 0.95]
    accuracies = []

    max_pred = 0
    max_thresh = 0
    for t in thresholds:
        preds = (all_probs > t).astype(float)
        acc = np.mean(all_labels == preds)
        if acc > max_pred: 
            max_pred = acc
            max_thresh = t
        accuracies.append(acc)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, accuracies, marker='o')
    plt.title("Test Accuracy vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("./accuracy_thresholds.png")
    plt.close()

    print(f"Max accuracy: {max_pred}, with threshold: {max_thresh}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # roc curve
    # precision recall curve


# MODEL_FILENAME = "./drive/MyDrive/Colab Notebooks/20Chan_bceloss_100_SMALL.pthE95"
MODEL_FILENAME = f"./final_runs/100E_20C_BCE_augment_clip_120.pthE68"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = ConvNeXt(**MODEL_CONFIG).to(device)
if COMPILE:
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))

eval_accuracy(model, device)
