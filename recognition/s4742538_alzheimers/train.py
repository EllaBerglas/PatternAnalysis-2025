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
from torch.optim.lr_scheduler import CosineAnnealingLR #type: ignore
from parameters import MODEL_FILENAME, MODEL_CONFIG, LEARNING_RATE, WEIGHT_DECAY, EPOCHS

# set gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# set up model
model = ConvNeXt(**MODEL_CONFIG).to(device)

if hasattr(torch, 'compile'):
    model = torch.compile(model)

# set up loss function and optimiser
#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

train_losses = []
val_losses = []

train_accs =  []
val_accs = []

# iterate over each epoch
for epoch in tqdm(range(EPOCHS)):
    model.train()
    epoch_train_loss = 0.0
    correct = 0
    total = 0

    # training on train set
    for batch_id, (image, label) in tqdm(enumerate(train_loader), desc = "Training"):
        image = image.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()  # backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_train_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    train_acc = correct / total
    train_accs.append(train_acc)

    epoch_val_loss = 0.0
    correct = 0
    total = 0
    model.eval()

    # evaluate with validation set after every epoch
    with torch.no_grad():
        for batch_id, (image, label) in enumerate(val_loader):

            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)

            epoch_val_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
            
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    val_acc = correct / total
    val_accs.append(val_acc)

    test_losses = []
    test_accs = []

    # evaluate with validation set after every epoch
    with torch.no_grad():
        for batch_id, (image, label) in enumerate(test_loader):

            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)

            epoch_test_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
            
    avg_test_loss = epoch_test_loss / len(val_loader)
    test_losses.append(avg_test_loss)

    test_acc = correct / total
    test_accs.append(test_acc)

    print(f"Epoch:{epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Test Acc: {test_acc:.3f}")

torch.save(model.state_dict(), MODEL_FILENAME)
print("model saved")
    
"""
Reload with:
model.load_state_dict(torch.load("convnext_alzheimer.pth"))
model.eval()
"""