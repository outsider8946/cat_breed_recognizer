import torch.nn as nn
class CatModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.regressor = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout(0.3),

        nn.Conv2d(32,32,kernel_size=(3,3),stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Flatten(),

        nn.Linear(8*8*32, 8*32),
        nn.ReLU(),

        nn.Linear(8*32, 4*32),
        nn.ReLU(),

        nn.Linear(4*32, 32),
        nn.ReLU(),

        nn.Linear(32, 4),
     )

    self.classifier = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout(0.3),

        nn.Conv2d(32,32,kernel_size=(3,3),stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Flatten(),

        nn.Linear(8*8*32, 4*32),
        nn.ReLU(),

        nn.Linear(4*32, 32),
        nn.ReLU(),

        nn.Linear(32, 6),
        nn.Softmax()
        #nn.BatchNorm1d(6)
      )

  def forward(self,x):
    # reshape
    x = x.view(x.shape[0], 3, 32,32)
    return self.regressor(x), self.classifier(x)
