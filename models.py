import torch.nn as nn
import torch.nn.functional  as F

class SingleLayerNN(nn.Module):
  def __init__(self, input_size, hidden_size, n_classes):
    super(SingleLayerNN, self).__init__()
    self.FC1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.FC2 = nn.Linear(hidden_size, n_classes)

  def forward(self, X):
    Z1 = self.FC1(X)
    A1 = self.relu(Z1)
    Z2 = self.FC2(A1)
    A2 = F.softmax(Z2, dim=1)
    return A2


class SoftmaxRegression(nn.Module):
    def __init__(self, input_size, n_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_size, n_classes)

    def forward(self, X):
      Z =  self.linear(X)
      A = F.softmax(Z,dim=1)
      return A