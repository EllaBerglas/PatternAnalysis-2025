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
from parameters import MODEL_CONFIG, COMPILE, MODEL_FILENAME, THRESHOLD
from sklearn.metrics import (confusion_matrix,  # type: ignore
                             classification_report,
                             ConfusionMatrixDisplay, 
                             roc_curve, auc ) # type: ignore

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

    # Plot confusion matrix as an image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig("./confusion_matrix.png", bbox_inches="tight")
    plt.close()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("./roc_curve.png", bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve as 'roc_curve.png' (AUC={roc_auc:.4f})")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = ConvNeXt(**MODEL_CONFIG).to(device)
if COMPILE:
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))

eval_accuracy(model, device)
