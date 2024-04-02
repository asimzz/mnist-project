import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from dataset import load_dataset
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

def load_dataset(data_origin):
    train_dataset = pd.read_csv(f"{data_origin}/mnist_train.csv",header=None)
    test_dataset = pd.read_csv(f"{data_origin}/mnist_test.csv",header=None)
    return train_dataset, test_dataset

class MNISTDataset(Dataset):

  input_size = 784
  n_classes = 5

  def __init__(self, dataset, indicies):
    targets = dataset[0]
    images = dataset.iloc[:, 1:]
    self.data = torch.tensor(images.loc[indicies].values, dtype=torch.float32) / 255
    self.targets = torch.tensor(targets.loc[indicies].values, dtype=torch.float32).reshape(-1,1)

    encoder.fit(self.targets)
    self.labels = encoder.categories_
    self.targets = torch.tensor(encoder.transform(self.targets)).to(torch.float32)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]


  def __len__(self):
    return self.data.shape[0]

