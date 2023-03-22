#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('train.csv')
print(df.info())
print(df.head())


# # 新增區段

# In[2]:


import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import imageio
import torch.optim as optim
from tqdm import tqdm


import matplotlib.pyplot as plt
import matplotlib
from model import VGG11

#from test import VGG
matplotlib.style.use('ggplot')


# # 新增區段

# In[3]:


# read the data
df_train = pd.read_csv('train.csv')
df_val = pd.read_csv('val.csv')
# get the image pixel values and labels
train_labels = df_train.iloc[0:,1]
train_images = df_train.iloc[:,0]
val_labels = df_val.iloc[0:, 1]
val_images = df_val.iloc[:,0]


# # 新增區段

# In[4]:


train_labels
#print(val_labels.shape)


# In[5]:


val_images.head()
print(val_images.shape)


# In[6]:


# define transforms
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))
])


# In[7]:


# our transforms will differ a bit from the VGG paper
# as we are using the MNIST dataset, so, we will directly resize...
# ... the images to 224x224 and not crop them and we will not use...
# ... any random flippings also
train_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5), std=(0.5))])
valid_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5), std=(0.5))])


# In[8]:


class VggLoader(data.Dataset):
  def __init__(self,mode, images, labels=None, transforms=None):
    self.mode = mode
    self.images_name = images
    self.label = labels
    self.transforms = transforms
  def __len__(self):
    return (len(self.images_name))
  def __getitem__(self, index):
    image_path = self.mode+'/'+self.images_name[index]
    self.img = imageio.imread(image_path)
    self.target = self.label[index]

    if self.transforms:
      self.img = self.transforms(self.img)
    return self.img , self.target


# In[9]:


train_dataset = VggLoader('train',train_images,train_labels,transform)
print(train_dataset[310])
valid_dataset = VggLoader('val',val_images,val_labels,transform)
#test_dataset = VggLoader('test')

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
#testloader = DataLoader(test_data, batch_size=64, shuffle=True)


# In[10]:


for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


# In[17]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO]: Computation device: {device}")
epochs = 77


# In[18]:


vgg_cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# In[19]:


# instantiate the model
model = VGG11(in_channels=3, num_classes=10).to(device)
#model = VGG(in_channels=3, num_classes=10, config=vgg_cfgs['A']).to(device)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"[INFO]: {total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO]: {total_trainable_params:,} trainable parameters.")
# the loss function
criterion = nn.CrossEntropyLoss()
# the optimizer
#optimizer=optim.Adam(model.parameters(),lr=LR,betas=(0.9,0.99)) 
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      )


# train

# In[20]:


# training
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1

        image, labels = data
        image = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# valid

# In[21]:


# validation
def validate(model, testloader, criterion):
    model.eval()

    # we need two lists to keep track of class-wise accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            # calculate the accuracy for each class
            correct  = (preds == labels).squeeze()
            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    # print the accuracy for each class after evey epoch
    # the values should increase as the training goes on
    print('\n')
    for i in range(10):
        print(f"Accuracy of digit {i}: {100*class_correct[i]/class_total[i]}")
    return epoch_loss, epoch_acc


# start train

# In[22]:


# start the training
# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")

    train_epoch_loss, train_epoch_acc = train(model, train_loader,optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)

    print('\n')
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

    print('-'*50)


# In[23]:


#plot



x1 = range(0, epoch+1)
x2 = range(0, epoch+1)
y1 = train_acc
y2 = train_loss
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Train accuracy vs. epoches')
plt.ylabel('Train accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.show()
plt.savefig("accuracy_loss.jpg")



x1 = range(0, epoch+1)
x2 = range(0, epoch+1)
y1 = valid_acc
y2 = valid_loss
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Valid accuracy vs. epoches')
plt.ylabel('Valid accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Valid loss vs. epoches')
plt.ylabel('Valid loss')
plt.show()
plt.savefig("accuracy_loss.jpg")


# save

# In[43]:


#FILE = 'model_acc_55.pt'
#torch.save(model, FILE)


# In[44]:


FILE = 'HW1_310581044.pt'
torch.save(model.state_dict(), FILE)


# In[ ]:




