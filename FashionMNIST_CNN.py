# %%

### Section 0 - Importing Dependencies

# Importing base PyTorch Library modules
import torch
from torch import nn

# Importing Torchvision modules
import torchvision

# Torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

# Importing Numpy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Importing helper functions
from helper_functions import print_train_time, train_test_model, eval_model

print(torch.__version__)
print(torchvision.__version__)

# Device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#%%
!nvidia-smi

# %%
### Section 1 - Importing FashionMNIST Data

# Importing the data from torchvision.datasets
train_data = torchvision.datasets.FashionMNIST(root="data",
                                               train=True,
                                               download=True,
                                               transform=torchvision.transforms.ToTensor(),
                                               target_transform=None)
                                            
test_data = torchvision.datasets.FashionMNIST(root="data",
                                              download=True,
                                              train=False,
                                              transform=torchvision.transforms.ToTensor(),
                                              target_transform=None)

# %%
print(f"Number of samples in Training Data: {len(train_data)}")
print(f"Number of samples in Testing Data: {len(test_data)}")

# %% Creating Dataloaders
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

batch_size = 32

train_dl = DataLoader(dataset=train_data,
                      batch_size=32,
                      shuffle=True,
                      collate_fn = lambda x:(p.to(device) for p in default_collate(x)))

test_dl = DataLoader(dataset=test_data,
                      batch_size=32,
                      shuffle=False,
                      collate_fn = lambda x:(p.to(device) for p in default_collate(x)))

# %%
print(f"Number of Training Batches: {len(train_dl)}")
print(f"Number of Test Batches: {len(test_dl)}")

print(f"Number of samples per Training Batch: {train_dl.batch_size}")
print(f"Number of samples per Test Batch: {test_dl.batch_size}")

class_names = train_data.classes

# %%

### Section 2 - Creating and training our FashionMNIST CNN

class FMNIST(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        
        self.conv1_block = nn.Sequential(nn.Conv2d(in_channels=input_shape,
                                                   out_channels=hidden_units,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(hidden_units),
                                        nn.Conv2d(in_channels=hidden_units,
                                                   out_channels=hidden_units,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(hidden_units),
                                        nn.MaxPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.Dropout(0.1)
                                        )
        
        self.conv2_block = nn.Sequential(nn.Conv2d(hidden_units, hidden_units*2,
                                                   3, padding=1),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.BatchNorm2d(hidden_units*2),
                                        nn.Conv2d(hidden_units*2, hidden_units*2,
                                                   3, padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(hidden_units*2),
                                        nn.MaxPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.Dropout(0.1)
                                        )
        
        self.fann = nn.Sequential(nn.Flatten(),
                                  nn.Linear(in_features=hidden_units*2*7*7,
                                            out_features=hidden_units*10),
                                  nn.ReLU(),
                                  nn.Linear(in_features=hidden_units*10,
                                            out_features=output_shape))
        
    def forward(self, x):
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.fann(x)
        return x

torch.manual_seed(42)
fmnist = FMNIST(1, 64, len(class_names)).to(device)
print(fmnist)


# %%
# Optimizer and Loss Function
optimizer = torch.optim.SGD(fmnist.parameters(),lr=0.1)
loss_fn = nn.CrossEntropyLoss().to(device)
acc_fn = Accuracy('multiclass', num_classes=10).to(device)


# %%
train_test_model(fmnist, loss_fn, optimizer, train_dl, test_dl, 
                 acc_fn, 5)



# %%
### Section 3 - Evaluating the performance of our CNN on Test Data
test_results = eval_model(fmnist, test_dl, loss_fn, len(test_data.classes), device)
print(test_results,"\n")

# Calculating predictions on the Whole Test Dataset (for Confusion Matrix)
from tqdm.auto import tqdm

test_preds = []

# To disable Dropout layers
fmnist.eval()

# To prevent gradient updation
with torch.inference_mode():
    for X, y in tqdm(test_dl):
        y_logits = torch.softmax(fmnist(X),dim=1)
        y_preds = torch.argmax(y_logits, dim=1)
        test_preds.append(y_preds.cpu())

# Concatenate list of predictions into a single 1D Tensor
test_preds = torch.cat(test_preds)


# %%
## Plotting Confusion Matrix on predictions for Test Data

from mlxtend.plotting import plot_confusion_matrix

conf_mat = ConfusionMatrix('multiclass', num_classes=len(test_data.classes))
# Here, torchvision's dataset.targets returns the label data
# The result of conf_mat will be a tensor -- change it to a Numpy Array
# so that we can plot it using mlxtend's plot_confusion_matrix
confmat_ny = conf_mat(test_preds, test_data.targets).numpy()

# Beautifying and plotting the Confusion Matrix
fig, ax = plot_confusion_matrix(conf_mat=confmat_ny,
                                class_names=test_data.classes,
                                figsize=(10,7),
                                cmap='cool')

# %%
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Here, .pth represents Pytorch
MODEL_NAME = "FashionMNIST_CNN_1.pth"
FINAL_MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#  Save the Model's State Dictionary, that contains the weights and other parameters.
print(f"Saving our final model to {FINAL_MODEL_SAVE_PATH}")
torch.save(obj=fmnist.state_dict(),f=FINAL_MODEL_SAVE_PATH)
print("Model Saved!")
