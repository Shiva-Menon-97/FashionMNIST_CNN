# FashionMNIST_CNN
 A CNN Architecture built on PyTorch to classify the Fashion MNIST Dataset.
 
 The CNN Architecture I implemented gave me a final Training Accuracy of 93% and a Testing Accuracy of 92%.

 Apart from performance boosts due to multiple Convolution Layers and Dropout Layers, this performance was achieved using Hyper-Parameter Optimization.
 
 Along with the Cross Entropy Loss function, I used the SGD optimizer with a fairly large learning rate of 0.15 over 5 epochs.
 
 I could use lr=0.15 since I was using Batch Normalization layers, which reduce Internal Covariance Shift in the distributions of the Inputs passing between the Convolutional Layers. This meant that we could accelerate the learning process without risking overshooting the optima.
 
 The CNN architecture is as follows:
 
  FMNIST:

  Convolutional Block 1: 

    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU()
    (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Dropout(p=0.1, inplace=False)
  
  Convolutional Block 2:

    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Dropout(p=0.1, inplace=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): ReLU()
    (6): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Dropout(p=0.1, inplace=False)
  
  Fully-Connected ANN:

    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=6272, out_features=640, bias=True)
    (2): ReLU()
    (3): Linear(in_features=640, out_features=10, bias=True)
  

  
  

