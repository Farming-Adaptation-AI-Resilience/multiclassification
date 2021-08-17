import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test/0000_ndvi_crop.png')
print(img)
print(np.unique(img))

img_ndvi = (img[:,:,0])/255
img_label = img_ndvi > 0.65
img_label = np.where(img_label==True,255,img_label)
img_label = np.where(img_label==False,0,img_label)
print(img_label)
print(img_label.shape)
print((np.dstack((img_label,img_label,img_label))).shape)
vis_img = np.dstack((img_label,img_label,img_label))
plt.imshow(vis_img)
plt.show()
cv2.imwrite('img.png',vis_img)
