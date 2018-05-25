

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import datetime
import argparse

with np.load('../data/prediction-challenge-02-data.npz') as fh:
    data_x = fh['data_x']
    data_y = fh['data_y']
    test_x = fh['test_x']

# TRAINING DATA: INPUT (x) AND OUTPUT (y)
# 1. INDEX: IMAGE SERIAL NUMBER (6000)
# 2. INDEX: COLOR CHANNELS (3)
# 3/4. INDEX: PIXEL VALUE (32 x 32)
print(data_x.shape, data_x.dtype)
print(data_y.shape, data_y.dtype)



# TEST DATA: INPUT (x) ONLY
print(test_x.shape, test_x.dtype)

# TRAIN MODEL ON data_x, data_y

training_x,validation_x=np.split(data_x,[5600])
training_y,validation_y=np.split(data_y,[5600])

print(training_x.shape)
print(training_y.shape)

print(validation_x.shape)
print(validation_y.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64, 64, 6)
        self.fc1 = nn.Linear(64*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.dropout = nn.Dropout(0.5)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = F.dropout(x, 0.4)
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = F.dropout(x, 0.3)
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.1)
        x = self.fc3(x)
        x=F.log_softmax(x,1)
        return x

def train(epoch,trainloader):
    print ("training epoch: ", epoch)
    running_loss=0.0
    for i, minibatch in enumerate(trainloader, 0):
        inputs, labels = minibatch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()   # zero the gradient buffers
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
    return

def test(validationloader):
    net.eval()
    correct = 0
   # with torch.no_grad():
    for data, target in validationloader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct_part = pred.eq(target.view_as(pred)).sum().item()
        correct+=correct_part
        print("correct part: ",correct_part*100./len(pred))

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(validationloader.dataset),
        100. * correct / len(validationloader.dataset)))
    return 100.*correct / len(validationloader.dataset)


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def reshuffle_sets(data_x,data_y):
    data_x,data_y=unison_shuffled_copies(data_x,data_y)
    training_x, validation_x = np.split(data_x, [5600])
    training_y, validation_y = np.split(data_y, [5600])
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(training_x).float(),torch.from_numpy(training_y))
    validationset=torch.utils.data.TensorDataset(torch.from_numpy(validation_x).float(),torch.from_numpy(validation_y))
    return trainset,validationset


net=Net().to(device)

# optimizer = torch.optim.SGD(net.param    added zero in lr
# optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
trainset,validationset=reshuffle_sets(data_x,data_y)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=30, shuffle=True)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=100, shuffle=True)
latest=0
i=0
while latest<77:
     train(i,trainloader)
     latest=int(test(validationloader))
     i+=1

# PREDICT prediction FROM test_x
tensorobj = torch.from_numpy(test_x).float()  # convert to tensor
tensorobj = tensorobj.to(device)

prediction = net.forward(tensorobj).detach().numpy()  # forward pass alle Bilder in testset
print("before argmax: ", prediction.shape)
prediction = np.argmax(prediction, 1)
print("after argmax: ", prediction.shape)



# MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
assert prediction.ndim == 1
assert prediction.shape[0] == 300
#
# # AND SAVE EXACTLY AS SHOWN BELOW

np.save('../data/prediction_'+str(datetime.datetime.now())+"_"+str(latest)+".npy", prediction)
