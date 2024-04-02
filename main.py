from dataset import load_dataset, MNISTDataset
from models import SoftmaxRegression, SingleLayerNN
from helpers import train_model

import torch
import torch.nn as nn
import torch.optim as optim


data_origin = '/content/drive/MyDrive/AMMI/MNIST_CSV'

train_dataset, test_dataset = load_dataset(data_origin)


def get_indicies(targets):
  return targets % 2 != 0, targets % 2 == 0

train_indices = get_indicies(train_dataset[0])
test_indices = get_indicies(test_dataset[0])

odd_train = MNISTDataset(dataset = train_dataset, indicies = train_indices[0])
even_train = MNISTDataset(dataset = train_dataset, indicies = train_indices[1])

odd_test = MNISTDataset(dataset = test_dataset, indicies = test_indices[0])
even_test = MNISTDataset(dataset = test_dataset, indicies = test_indices[1])

odd_classifier = SoftmaxRegression(MNISTDataset.input_size, MNISTDataset.n_classes)
even_classifier = SoftmaxRegression(MNISTDataset.input_size, MNISTDataset.n_classes)

criterion = nn.CrossEntropyLoss()
odd_optimizer = optim.Adam(odd_classifier.parameters(), lr=0.01)
even_optimizer = optim.Adam(even_classifier.parameters(), lr=0.01)

train_model(odd_train,odd_test, odd_classifier, odd_optimizer)

train_model(even_train, even_test, even_classifier, even_optimizer)


odd_model = SingleLayerNN(MNISTDataset.input_size, 500, MNISTDataset.n_classes)
even_model = SingleLayerNN(MNISTDataset.input_size, 500, MNISTDataset.n_classes)

odd_optimizer = optim.Adam(odd_model.parameters(), lr=0.01)
even_optimizer = optim.Adam(even_model.parameters(), lr=0.01)

train_model(odd_train, odd_test, odd_model, odd_optimizer)

train_model(even_train, even_test, even_model, even_optimizer)



"TRANSFER LEARNING - EVEN NUMBERS USING ODD MODEL"


layer1_weight = odd_model.FC1.weight.data

X_train = (nn.ReLU()((layer1_weight @ even_train.data.T))).T
X_test = (nn.ReLU()((layer1_weight @ even_test.data.T))).T

n_samples, n_features = X_train.shape
n_classes = 5
trans_even_model = SoftmaxRegression(n_features, n_classes)
n_epochs = 1000
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(trans_even_model.parameters(), lr=lr)
train_model(X_train, even_train.targets, X_test, even_test.targets, optimizer, criterion, trans_even_model, n_epochs)

"TRANSFER LEARNING - ODD NUMBERS USING EVEN MODEL"

layer1_weight = even_model.FC1.weight.data

X_train = (nn.ReLU()((layer1_weight @ odd_train.data.T))).T
X_test = (nn.ReLU()((layer1_weight @ odd_test.data.T))).T

n_samples, n_features = X_train.shape
n_classes = 5
trans_odd_model = SoftmaxRegression(n_features, n_classes)
n_epochs = 1000
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(trans_odd_model.parameters(), lr=lr)
train_model(X_train, odd_train.targets, X_test, odd_test.targets, optimizer, criterion, trans_odd_model, n_epochs)