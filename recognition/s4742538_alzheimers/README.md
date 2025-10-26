# TITLE

## Description of Algorithm and the problem that it solves
One paragraph


## How it works
in a paragraph


**Must include figure or visualisation**

## List of dependencies 
with versions 
and address reproduciblility of results



## project specific stuff?

## Example inputs, outputs and plots of your algorithm


## Describe any specific pre-processing you have used with references if any. Justify your training, validation
## and testing splits of the data.


# evaluations
false negatives are bad, test on each alzheimers and normal to determine tp, fp etc
medical diagnosis do not want to under diagnose. \



# Log
First started by passing in just the images (and accidentally didnt seperate out the validation images properly whoops). Resuslted in high accuracy for validation data (99.7% ish) and low for the training data (67%). This told me there was data leakage. 
Seperating by person_id meant that validation accuracy was more representative of the prediction accuracy.
However, I thought that maybe providing the persons id could help the model use multiple images to determine alzheimers.

Secondly, I still haddn't changed anything about the training process and resulted in bad results.
There are a few different ways I could go about trying to increase the accuracy.
1. use a bigger model.
    I was already usign a decently sized model, usign a bigger one would take up more resources, so I will hold off on that one
2. Play with parameters
    this is more so for fine tuning, there seems to be something significantly wrong with the model
3. Data does not have enough information, adjust the information being input
    I figured that a group of the persons images would contain better quality information than being spread out
    So I changed the input data to the model. 


Could try 3D (putting the sliced back together) but it is heavy on memory and would be better if I still had the OG 3D data. Plus doesn't quite fir the task desciption.
So, at least to start, I am going to aggregate the slices per person together

Another thing I could do:
Train a 2D CNN to extract features from each slice, Aggregate all slice features (mean, max, or an RNN/attention), then Feed that into a final classifier for the subject


Unfortunatly this did not help that much

for some reason, test and validation results seem quite different. 
I get train acc 1.0, val acc 0.75 and test acc 0.58
Although this is obviously overfitting, I do not understand why there is a significant difference between the validation and test result accuracy. This suggests some sort of data leakage or significant difference of the test set to the training set.


Now added the following:
a different train transformer:
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.Grayscale(num_output_channels=CHANNELS),
    # Add aggressive augmentations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), #translates and zoom up to 10%
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1), # random ajustments to brightness
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)), # randomly crops a little bit
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))  # Random occlusion
])
 label smoothing to criteria
 criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

A scheduelr (not much difference on its own from what I can tell)

"drop_path_rate": 0.2  # dropout for residual connections (on colab)
