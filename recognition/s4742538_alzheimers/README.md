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

