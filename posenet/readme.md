# Including pose errors
Since pose errors require extra information about each image (the ground truth pose) we need to make a custom data holder for the data API with IQM-VIS.

This is found in `data_holder.py`, which also does all the posenet network loading.

## Extra dependancies
requires `torchvision` and posenet model weights
