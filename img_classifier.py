import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import time
import torch
import cv2
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
def input_p(img):
    img = img.float()   
    img = img[None, ...]
    # img = img.permute(0,3,2,1)
    return img 
def our_image_classifier(image):
    '''
            Function that takes the path of the image as input and returns the closest predicted label as output
            '''
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = {0:'Septoria' ,
         1:'Powdery Mildew',
         2:'Healthy' ,
         3:'Tobacco Mosiac Virus',
         4: 'Spider Mites',
         5:'Calcium Deficiency' ,
         6:'Magnesium Deficiency' }

    model = torch.load('/gdrive/MyDrive/Final_Project/Project_Code/model_checkpoints/resnet50_epoch35')
    #st.error("after loading the model")
    #img = Image.open("/content/img_ai_app_boilerplate/model/H307_2.jpg")
    # open_cv_image = np.array(image) 
    # # Convert RGB to BGR 
    # img = open_cv_image[:, :, ::-1].copy() 
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(),
    ToTensorV2(),
    ])
    # img = torch.from_numpy(img)
    img = transform(image=img)["image"]
    img = input_p(img)

    img = img.to(device)
    # st.error("img to device")
    output = model(img)
    # st.error("model")
    m = nn.Sigmoid()  
    # st.error("sigmoid")
    pred = m(output)
    # st.error("after labeling")
    pred_max = torch.argmax(pred, dim = 1)
    label_id = pred_max.tolist()[0]
    # st.error(label_id) 
    label = labels[label_id]
    return label_id
