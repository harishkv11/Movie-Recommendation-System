# Downloading the dataset and unzipping it
!wget "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
!unzip ml-100k.zip
!ls

!wget "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
!unzip ml-1m.zip
!ls

# Importing the libraries
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Extracting Useful Data
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
train = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
train = np.array(train, dtype = 'int')
test = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test = np.array(test, dtype = 'int')

# Getting the number of users and movies
no_of_users = int(max(max(train[:, 0], ), max(test[:, 0])))
no_of_movies = int(max(max(train[:, 1], ), max(test[:, 1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
  new_data = []
  for ids in range(1, no_of_users + 1):
    id_movies = data[:, 1] [data[:, 0] == ids]
    id_ratings = data[:, 2] [data[:, 0] == ids]
    ratings = np.zeros(no_of_movies)
    ratings[id_movies - 1] = id_ratings
    new_data.append(list(ratings))
  return new_data
train = convert(train)
test = convert(test)

# Converting the data into Torch tensors
train = torch.FloatTensor(train)
test = torch.FloatTensor(test)

# Architecture of the Neural Network
class model(nn.Module):
    def __init__(self, ):
        super(model, self).__init__()
        self.fc1 = nn.Linear(no_of_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, no_of_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = model()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200
epoch_list = []
loss_list = []
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id in range(no_of_users):
    input = Variable(train[id]).unsqueeze(0)
    target = input.clone()
    if torch.sum(target.data > 0) > 0:
      output = sae(input)
      target.require_grad = False
      output[target == 0] = 0
      loss = criterion(output, target)
      mean_corrector = no_of_movies/float(torch.sum(target.data > 0) + 1e-10)
      loss.backward()
      train_loss += np.sqrt(loss.data*mean_corrector)
      s += 1.
      optimizer.step()
  epoch_list.append(int(epoch))
  loss_list.append(float(train_loss/s))

# Relation of Training Loss with Epoch
plt.plot(epoch_list,loss_list)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.show()

# Testing the SAE
test_loss = 0
s = 0.
for id in range(no_of_users):
  input = Variable(train[id]).unsqueeze(0)
  target = Variable(test[id]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = no_of_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.

# Final Result
print('Final Test Loss: '+str(float(test_loss/s)))