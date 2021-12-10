import torch
import torch.optim as optim
import torch.nn as nn
import preprocessor as pp

def accuracy(outputs, label):
    if (label - outputs) < 0.5:
        return 1
    return 0

def training_testing_data(trainloader, testloader, num_of_epoches):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    running_loss = 0.0
    running_acc = 0.0
    iter_num = 0.0
    for epoch in range(num_of_epoches):
        net = LeNet()
        net = net.float()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            outputs = net(inputs.float())
            loss = nn.functional.binary_cross_entropy(outputs[0], labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += accuracy (outputs, labels)
            iter_num += 1
        train_loss.append(running_loss/iter_num)
        train_acc.append(running_acc/iter_num)
        running_loss = running_acc = iter_num = 0.0



        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            outputs = net(inputs.float())
            loss = nn.functional.binary_cross_entropy(outputs[0], labels.float())
            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)
            iter_num += 1
            # if i % 2000 == 1999:
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
        test_loss.append(running_loss / iter_num)
        test_acc.append(running_acc / iter_num)
        running_loss = running_acc = iter_num = 0.0


    return train_loss, train_acc, test_loss, test_acc



class LeNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=66, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=66, out_channels=216, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=216, out_channels=66, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=66, out_channels=6, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(24, 66),  # Why 16*5*5 ?
            nn.ReLU(),
            nn.Linear(66, 216),
            nn.ReLU(),
            nn.Linear(216, 66),
            nn.ReLU(),
            nn.Linear(66, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.ReLU(),
            nn.Sigmoid(),

        )
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_scores = self.classifier(features)
        return class_scores

