"""
showing example usage of your trained model. Print out any results and / or provide 
visualisations where applicable
"""
from collections import defaultdict
import os
from dataset import test_loader
import torch # type: ignore
#from parameters import MODEL_FILENAME
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
from modules import ConvNeXt
from parameters import MODEL_CONFIG, COMPILE
from sklearn.metrics import confusion_matrix, classification_report  # type: ignore

def eval_accuracy(model, device):
    all_labels = []
    all_preds = []
    model.eval()
    print(f"Loaded model")
    class_names = ["AD", "NC"]

    with torch.no_grad():
        for (volumes, labels) in tqdm(test_loader, desc="Test Set"):
            volumes = volumes.to(device) # volumes: (B, 20, H, W)
            labels = labels.to(device)

            outputs = model(volumes)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    test_accuracy = np.mean(all_labels == all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))



def main():
    MODEL_FILENAME = f"./models/100E_2_weights/20Chan_100_SMALL.pthE94"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model = ConvNeXt(**MODEL_CONFIG).to(device)
    if COMPILE:
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))

    eval_accuracy(model, device)


if __name__ == "__main__":
    main()