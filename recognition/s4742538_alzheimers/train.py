"""
containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""
import torch  # type: ignore
from modules import ConvNeXt
from dataset import test_loader, train_loader, val_loader
from torch import optim, nn  #type: ignore
from tqdm import tqdm #type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 10

model = ConvNeXt(
    in_chans=1, 
    num_classes=2, 
    depths=[3, 3, 9, 3], 
    dims=[96, 192, 384, 768]
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 

train_losses = []
train_dcs = []
val_losses = []
val_dcs = []

for epoch in tqdm(range(EPOCHS)):
    model.train()

    # training
    for image, label in enumerate(tqdm(train_loader, position=0, leave=True)):
        image.to_device(device)
        label.to_device(device)
        
        output = model(image)
        loss = criterion(output, label)
        loss.backward()  # backpropegation
        optimizer.step()

        train_running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        for image, label in enumerate(tqdm(val_loader, position=0, leave=True)):
            image.to_device(device)
            label.to_device(device)

            output = model(image)
            loss = criterion(output, label)

            val_running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")