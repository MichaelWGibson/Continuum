from Data_Aug.aug_fun import *
from Data_Aug.bbox_util import *
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import pickle as pkl
get_ipython().run_line_magic('matplotlib', 'inline')

imageName = "frame-399.jpg"
fileName = imageName[:imageName.find('.jpg')] 3 cuts of .jpg so we can save new photos with same name

# loads image
img = cv2.imread(imageName)[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb

#This takes the coordinates of the top left and bottom right of the bounding boxes and saves it as an np.array
bboxes = np.array([[261.389, 299.722, (261.389+33.444), (299.722+30.056), 1],
                  [221.333, 284.889, (221.333+32.889), 284.889+26.111, 1]])


#draws rectangles over original image
plotted_img = draw_rect(img, bboxes)
cv2.imwrite("NewPhotos/" + fileName + ".jpg", plotted_img)

# flips the image horizontally
img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_1.jpg", plotted_img)

#randomly scales image in or out
img_, bboxes_ = RandomScale(0.3, diff = True)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_2.jpg", plotted_img)

#Translate image in random direction
img_, bboxes_ = RandomTranslate(0.3, diff = True)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_3.jpg", plotted_img)

# Randomly rotates image
img_, bboxes_ = RandomRotate(20)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_4.jpg", plotted_img)

# Randomly shears image
img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_5.jpg", plotted_img)

#Resizes image
img_, bboxes_ = Resize(608)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_6.jpg", plotted_img)

#Adjusts brightness to be brighter
img_, bboxes_ = RandomBrightnessUp(60)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_7.jpg", plotted_img)

#Adjusts brightness to be darker
img_, bboxes_ = RandomBrightnessDown(40)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_8.jpg", plotted_img)

# random sequance of the above augmentations
seq = Sequence([RandomBrightnessUp(60),RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear()])
img_, bboxes_ = seq(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_9.jpg", plotted_img)

# random sequance of the above augmentations
seq = Sequence([RandomHorizontalFlip(), RandomTranslate(), RandomShear()])
img_, bboxes_ = seq(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_10.jpg", plotted_img)

# random sequance of the above augmentations
seq = Sequence([RandomScale(), RandomRotate(20), RandomShear()])
img_, bboxes_ = seq(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("NewPhotos/" + fileName + "_11.jpg", plotted_img)

