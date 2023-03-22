#!/usr/bin/env python
# coding: utf-8

# In[39]:


import torch
import cv2
import glob as glob
import torchvision.transforms as transforms
import numpy as np
from model import VGG11
import imageio


# In[40]:


# inferencing on CPU
device = 'cpu'
# initialize the VGG model

model = VGG11(in_channels=3, num_classes=10)
#model = torch.load('model_acc_51.pt')
model.load_state_dict(torch.load('HW1_310581044.pt'))
model.to(device)


model.eval()
# simple image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
])


# In[41]:


num,inference_label =[],[]

# get all the test images path
image_paths = glob.glob('test/*.jpg')
for i, image_path in enumerate(image_paths):
    image=imageio.imread(image_path)
    #image= cv2.imread(image_path)
    id=image_path
    id_list = list(id)
    for i in range(5):
        id_list.pop(0)
    id_final = ''.join(id_list)
    #print(id_final)
    num.append(id_final)
    

    image = transform(image)
    # add one extra batch dimension
    image = image.unsqueeze(0).to(device)
    # forward pass the image through the model
    outputs = model(image)
    # get the index of the highest score
    label = np.array(outputs.detach()).argmax()
    inference_label.append(label)
    print(label)


# In[42]:


import pandas as pd
output =pd.DataFrame({'names':num,'label':inference_label})


# In[43]:


output.head()


# In[45]:


output.to_csv("HW1_310581044.csv",index=False)


# In[ ]:




