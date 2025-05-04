import cv2
import numpy as np
from sklearn import metrics

def get_images(path, classes):
  images_path = f"{path}/{classes}.png"
  images = cv2.imread(images_path, cv2.IMREAD_UNCHANGED)
  return images

def get_overlay(bg_image, fg_image, size = (60,60)):
  fg_image = cv2.resize(fg_image, size)  
  
  fg_mask = fg_image[:,:,3:]
  fg_image = fg_image[:, :, :3]
  bg_mask = 255 - fg_mask
  bg_image = bg_image / 255
  fg_image = fg_image / 255
  fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) / 255
  bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR) / 255
  image = cv2.addWeighted(bg_image*bg_mask, 255, fg_image*fg_mask, 255, 0.).astype(np.uint8)

  return image