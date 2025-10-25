"""
showing example usage of your trained model. Print out any results and / or provide 
visualisations where applicable
"""

from dataset import test_loader
import torch # type: ignore
from parameters import MODEL_FILENAME
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
from modules import ConvNeXt
from parameters import MODEL_CONFIG


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

all_labels = []
all_preds = []

model = ConvNeXt(**MODEL_CONFIG).to(device)
model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
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