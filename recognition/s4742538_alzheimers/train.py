"""
containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""
import torch  # type: ignore
from modules import ConvNeXt
from dataset import train_loader, val_loader, test_loader
from torch import optim, nn  #type: ignore
from tqdm import tqdm #type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts #type: ignore
from parameters import MODEL_FILENAME, MODEL_CONFIG, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, COMPILE

import matplotlib # type: ignore 
matplotlib.use("Agg") # to work in wsl (no ability to display)
import matplotlib.pyplot as plt  # type: ignore

# set gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# set up model
model = ConvNeXt(**MODEL_CONFIG).to(device)

if COMPILE:
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

# set up loss function, optimiser and scheduler

#criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# penalises incorrect AD predictions a bit more
weighted_penalty = torch.tensor([1.3], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=weighted_penalty)
optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#scheduler = CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=1e-6) # try next
scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=10, T_mult=2)

train_losses = []
train_accs =  []

val_losses = []
val_accs = []

# for saving the best model
best_val_acc = 0.0
best_val_loss = float("inf")
best_epoch = -1

# iterate over each epoch
for epoch in tqdm(range(EPOCHS)):
    model.train()
    epoch_train_loss = 0.0
    correct = 0
    total = 0

    # training on train set
    for batch_id, (image, label) in tqdm(enumerate(train_loader), desc = "Training"):
        image = image.to(device)
        label = label.to(device).float().unsqueeze(1)
        
        optimiser.zero_grad()
        output = model(image)
        loss = criterion(output, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        epoch_train_loss += loss.item()
        predicted = (torch.sigmoid(output) > 0.5).float()
        correct += (predicted == label).sum().item()
        total += label.size(0)

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    train_acc = correct / total
    train_accs.append(train_acc)

    epoch_val_loss = 0.0
    val_correct = 0
    val_total = 0
    model.eval()

    # evaluate with validation set after every epoch
    with torch.no_grad():
        for batch_id, (image, label) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device).float().unsqueeze(1)
            # label = label.to(device)

            output = model(image)
            loss = criterion(output, label)

            epoch_val_loss += loss.item()
            #_, predicted = torch.max(output, 1)
            predicted = (torch.sigmoid(output) > 0.5).float()
            val_correct += (predicted == label).sum().item()
            val_total += label.size(0)
            
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    val_acc = val_correct / val_total
    val_accs.append(val_acc)

    scheduler.step(avg_val_loss)
    current_lr = optimiser.param_groups[0]['lr']

    # check if this is the best model so far
    if val_acc >= best_val_acc:  
        best_val_acc = val_acc
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), (MODEL_FILENAME + f"E{epoch}"))
        print(f"model saved E {best_epoch}, best_val_acc: {best_val_acc:.3f}")
    
    print(f"Epoch:{epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f},  Val Loss: {avg_val_loss:.4f}, \
          Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

    
"""Plot accuracies and loss from this training"""
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(12, 5))

# loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label='Train Accuracy', marker='o')
plt.plot(epochs_range, val_accs, label='Validation Accuracy', marker='o')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# save plots
plt.tight_layout()
plt.savefig("training_accuracy_loss.png")
plt.close()
