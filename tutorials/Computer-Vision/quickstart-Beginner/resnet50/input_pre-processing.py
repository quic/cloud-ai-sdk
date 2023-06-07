###############################################################################
#  Convert the test image to the tensor format expected by the ResNet-50 model. 
###############################################################################

#!/bin/bash

import numpy as np
import cv2

def image_to_tensor(imagepath, dims, nchw):
  img = cv2.imread(imagepath, cv2.IMREAD_COLOR)  #bgr
  img = cv2.resize(np.array(img), dims,
                   interpolation=cv2.INTER_LINEAR).astype(np.float32)  #bgr
  imdata = np.asarray(img, dtype="float32")   #bgr

  if (nchw == True):
    # NHWC to NCHW
    imdata = imdata[:,:,0:3].transpose(2, 0, 1)  #bgr
    # Normalize
    imdata = np.asarray([(imdata[0,:,:]-102.255)*0.017429194,
                       (imdata[1,:,:]-116.28)*0.017507003,
                       (imdata[2,:,:]-123.675)*0.017124754],dtype="float32")
  else:
    imdata = np.asarray([(imdata[:,:,0]-103.94),
                       (imdata[:,:,1]-116.78),
                       (imdata[:,:,2]-123.68)],dtype="float32")
    imdata = np.reshape(imdata, (imdata.shape[1],imdata.shape[2],imdata.shape[0]))
    
  imdata = np.asarray([imdata], dtype="float32")
  return imdata
  
tensor = image_to_tensor('cat_281_299.png', (224, 224), True)
tensor.tofile('data.raw')
print("Pre-Processed data is created and saved as data.raw")

with open('image.lst', 'w') as f:
    f.write('./data.raw\n')
