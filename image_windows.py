import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from lesson_functions import *

image = mpimg.imread('data/test1.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, image.shape[0]//2], y_start_stop=[None, None], 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))
windows1 = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[image.shape[0]//2, (image.shape[0]//4)*3], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
windows2 = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[image.shape[0]//2, image.shape[0]], xy_window=(192, 192), xy_overlap=(0.70, 0.70))
windows3 = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[image.shape[0]//2, image.shape[0]], xy_window=(256, 256), xy_overlap=(0.75, 0.75))

windows = np.concatenate((windows, windows1))
windows = np.concatenate((windows, windows2))
windows = np.concatenate((windows, windows3))

print(windows)