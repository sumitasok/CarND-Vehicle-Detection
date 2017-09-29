# This is the main file to run

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from moviepy.editor import VideoFileClip
import pickle

# pickle_file = "input/LinearSVC_histbin_32_spacial_32_1506671036.pkl"
# pickle_file = "input/LinearSVC_histbin_32_spacial_32_2017Jul.pkl"
pickle_file = "input/LinearSVC_histbin_32_spacial_32_hog__2017Jul.pkl"
# pickle_file = "input/LinearSVC_histbin_32_spacial_32_1506673466.pkl"
# pickle_file = "input/LinearSVC_histbin_32_spacial_32_1506673702.pkl"
pickle_file = "input/LinearSVC_histbin_32_spacial_32_hog1506679413.pkl"
# http://scikit-learn.org/stable/modules/model_persistence.html
from sklearn.externals import joblib

data = joblib.load(pickle_file)

# print("svc :", data)

# svc = data

# data = dict()
# data['color_space'] = 'YCrCb'
# data['hist_bins'] = 32
# data['orient'] = 9
# data['pix_per_cell'] = 8
# data['cell_per_block'] = 2
# data['hog_channel'] = 'ALL'
# data['spatial_feat'] = False
# data['hist_feat'] = True
# data['hog_feat'] = True
# data['spatial_size'] = (16, 16)

svc = data['svc']
X_scaler = data['X_scaler']
color_space = data['color_space']
spatial_size = data['spatial_size']
hist_bins = data['hist_bins']
orient = data['orient']
pix_per_cell = data['pix_per_cell']
cell_per_block = data['cell_per_block']
hog_channel = data['hog_channel']
spatial_feat = data['spatial_feat']
hist_feat = data['hist_feat']
hog_feat = data['hog_feat']


image = mpimg.imread('input/test1.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, image.shape[1]], y_start_stop=[375, 700], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))
windows1 = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[375, 700],
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))
windows2 = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[375, 700],
                    xy_window=(100, 100), xy_overlap=(0.30, 0.30))
# windows3 = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[image.shape[0]//2, image.shape[0]],
#                   xy_window=(256, 256), xy_overlap=(0.75, 0.75))


for window in windows1:
    windows.append(window)

for window in windows2:
    windows.append(window)


def process_image(image):
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    # plt.imshow(window_img)
    # plt.savefig("input/hot.jpg")

    # Read in a pickle file with bboxes saved
    # Each item in the "all_bboxes" list will contain a 
    # list of boxes for one of the images shown above
    # box_list = pickle.load( open( "bbox_pickle.p", "rb" ))

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img


# white_output = 'input/car_detection.mp4'
# clip1 = VideoFileClip("input/test_video.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)
import scipy.misc
images = glob.glob('input/video/*.jpg')

for image in images:
    img = mpimg.imread(image)
    t3 = time.time()
    draw_image = process_image(img)
    t4 = time.time()
    print(round(t4-t3, 2), ' Seconds to process ', image)
    scipy.misc.imsave('input/images/'+image.split("/")[-1], draw_image)
# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(draw_img)
# plt.title('Window')
# plt.subplot(122)
# plt.imshow(heatmap, cmap='hot')
# plt.title('HOG')
# fig.tight_layout()
# plt.savefig("input/heat.jpg")

