import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
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

training_x,validation_x=np.split(data_x,[5500])
training_y,validation_y=np.split(data_y,[5500])

print(training_x.shape)
print(training_y.shape)

print(validation_x.shape)
print(validation_y.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch):
    print ("training epoch: ", epoch)
    running_loss=0.0
    for i, minibatch in enumerate(trainloader, 0):
        inputs,labels=minibatch
        optimizer.zero_grad()   # zero the gradient buffers
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
    return

def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validationloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(validationloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validationloader.dataset),
        100. * correct / len(validationloader.dataset)))


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

trainset = torch.utils.data.TensorDataset(torch.from_numpy(training_x),torch.from_numpy(training_y))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)

validationset=torch.utils.data.TensorDataset(torch.from_numpy(validation_x),torch.from_numpy(validation_y))
validationloader=torch.utils.data.DataLoader(validationset,batch_size=50,shuffle=False)

criterion = nn.MSELoss()
net=Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


for i in range(10):
     train(i)
     test()



# PREDICT prediction FROM test_x

# MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
# assert prediction.ndim == 1
# assert prediction.shape[0] == 300
#
# # AND SAVE EXACTLY AS SHOWN BELOW
# np.save('prediction.npy', prediction)
