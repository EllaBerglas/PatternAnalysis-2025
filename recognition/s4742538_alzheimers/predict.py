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
from parameters import MODEL_CONFIG


def eval_accuracy(model, device):
    all_labels = []
    all_preds = []
    model.eval()
    print(f"Loaded model: {MODEL_FILENAME}")

    with torch.no_grad():
        for (images, labels) in tqdm(test_loader, desc="Test Set"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    test_accuracy = np.mean(all_labels == all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")


"""
New one, groups the answers bet person and averages
"""

from collections import defaultdict
import os

def eval_accuracy_by_group(model, device):
    """
    Evaluate model accuracy for each person (averaged)
    """

    model.eval()
    person_to_probs = defaultdict(list)
    person_to_labels = {}

    dataset = test_loader.dataset

    if hasattr(dataset, "indices"): # handles both Subset or plain ImageFolder
        samples = [dataset.dataset.samples[i] for i in dataset.indices]
    else:
        samples = dataset.samples

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Grouped Eval")):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            # match batch to corresponding file paths
            batch_samples = samples[i * test_loader.batch_size : i * test_loader.batch_size + len(labels)]

            for (path, lable), prob in zip(batch_samples, probs):
                pid = os.path.basename(path).split("_")[0] 
                person_to_probs[pid].append(prob)
                person_to_labels.setdefault(pid, int(lable))
                if person_to_labels[pid] != int(lable):
                    print(f"inconsistent labels for {pid}")

    # aggregate probs for each person
    y_true, y_pred = [], []
    for pid, probs in person_to_probs.items():
        mean_prob = np.mean(probs, axis=0)
        pred = int(np.argmax(mean_prob))
        y_pred.append(pred)
        y_true.append(person_to_labels[pid])

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = np.mean(y_true == y_pred)

    print(f"Subject-level accuracy: {acc:.4f} ({len(y_true)} subjects)")
    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model = ConvNeXt(**MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))

    eval_accuracy(model, device)
    eval_accuracy_by_group(model, device)
    

if __name__ == "__main__":
    main()