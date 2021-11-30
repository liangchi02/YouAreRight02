```python
# 1.Import relevant package
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans
import time
```


```python
# Hyper parameters
BATCH_SIZE = 100
nepochs = 50
LR = 0.001

# definning loss function
loss_func = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```


```python
mean = [x/255 for x in [125.3, 23.0, 113.9]] 
std = [x/255 for x in [63.0, 62.1, 66.7]]
n_train_samples = 50000

if __name__ == '__main__':
    
    train_set = dsets.CIFAR10(root='CIFAR10/',
                              train=True,
                              download=True,
                              transform=trans.Compose([
                                 trans.RandomHorizontalFlip(),
                                 trans.RandomCrop(32, padding=4),
                                 trans.ToTensor(),
                                 trans.Normalize(mean, std)
                             ]))
    train_dl = DataLoader(train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=6)
    
    # train_set.train_data = train_set.train_data[0:n_train_samples]
    # train_set.train_labels = train_set.train_labels[0:n_train_samples]
    
    test_set = dsets.CIFAR10(root='CIFAR10/',
                             train=False,
                             download=True,
                             transform=trans.Compose([
                                trans.ToTensor(),
                                trans.Normalize(mean, std)
                            ]))

    test_dl = DataLoader(test_set,
                         batch_size=BATCH_SIZE,
                         num_workers=6)
```

    Files already downloaded and verified
    Files already downloaded and verified
    


```python
def eval(model, loss_func, dataloader):

    model.eval()
    loss, accuracy = 0, 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            error = loss_func(logits, batch_y)
            loss += error.item()

            probs, pred_y = logits.data.max(dim=1)
            accuracy += (pred_y==batch_y.data).float().sum()/batch_y.size(0)

    loss /= len(dataloader)
    accuracy = accuracy*100.0/len(dataloader)
    return loss, accuracy

def train_epoch(model, loss_func, optimizer, dataloader):

    model.train()
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        error = loss_func(logits, batch_y)
        error.backward()
        optimizer.step()
```


```python
# 定义卷积层，在VGGNet中，均使用3x3的卷积核
def conv3x3(in_features, out_features): 
    return nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
```


```python
# 搭建VGG19，除了卷积层外，还包括2个全连接层（fc_1、fc_2），1个softmax层
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 1.con1_1
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 2.con1_2
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3.con2_1
            conv3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4.con2_2
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 5.con3_1
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6.con3_2
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7.con3_3
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 8.con3_4
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 9.con4_1
            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 10.con4_2
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 11.con4_3
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 12.con4_4
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 13.con5_1
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14.con5_2
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 15.con5_3
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 16.con5_4
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )

        self.classifier = nn.Sequential(
            # 17.fc_1
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 18.fc_2
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 19.softmax
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
```

After defining VGG19 network，start trainning.


```python
vgg19 = VGG().to(device)

optimizer = torch.optim.Adam(vgg19.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
learn_history = []
```


```python
print('Start training VGG19……')

for epoch in range(nepochs):
    # Train start time
    since = time.time()
    train_epoch(vgg19, loss_func, optimizer, train_dl)
    
    # Output train result every 5 turns
    if (epoch)%5 == 0:
        tr_loss, tr_acc = eval(vgg19, loss_func, train_dl)
        te_loss, te_acc = eval(vgg19, loss_func, test_dl)
        learn_history.append((tr_loss, tr_acc, te_loss, te_acc))
        # Time for completing one batch train
        now = time.time()
        print('[%3d/%d, %.0f seconds]|\t Train error: %.1e, Train accuracy: %.2f\t |\t Train error: %.1e, Train accuracy: %.2f'%(
            epoch+1, nepochs, now-since, tr_loss, tr_acc, te_loss, te_acc))
```

    Start training VGG19……
    [  1/50, 1617 seconds]|	 Train error: 2.0e+00, Train accuracy: 22.55	 |	 Train error: 2.0e+00, Train accuracy: 22.81
    [  6/50, 1533 seconds]|	 Train error: 1.4e+00, Train accuracy: 53.45	 |	 Train error: 1.3e+00, Train accuracy: 53.42
    
