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

pickle_file = "data/LinearSVC_histbin_32_spacial_32_hog.pkl"
# http://scikit-learn.org/stable/modules/model_persistence.html
from sklearn.externals import joblib

data = joblib.load(pickle_file)

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


image = mpimg.imread('data/test1.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255



def process_image(image):
	# windows = slide_window(image, x_start_stop=[None, image.shape[0]//2], y_start_stop=[image.shape[0]//2, (image.shape[0]//4)*3], 
	#                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))
	windows = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[image.shape[0]//2, (image.shape[0]//4)*3], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
	# windows = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[image.shape[0]//2, image.shape[0]], xy_window=(192, 192), xy_overlap=(0.70, 0.70))
	# windows = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[image.shape[0]//2, image.shape[0]], xy_window=(256, 256), xy_overlap=(0.75, 0.75))


	# for window in windows1:
	#     windows.append(window)

	# for window in windows2:
	#     windows.append(window)

	# for window in windows3:
	#     windows.append(window)

	hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
	                        spatial_size=spatial_size, hist_bins=hist_bins, 
	                        orient=orient, pix_per_cell=pix_per_cell, 
	                        cell_per_block=cell_per_block, 
	                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
	                        hist_feat=hist_feat, hog_feat=hog_feat)                       

	window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

	# plt.imshow(window_img)
	# plt.savefig("data/hot.jpg")

	# Read in a pickle file with bboxes saved
	# Each item in the "all_bboxes" list will contain a 
	# list of boxes for one of the images shown above
	# box_list = pickle.load( open( "bbox_pickle.p", "rb" ))

	heat = np.zeros_like(image[:,:,0]).astype(np.float)

	# Add heat to each box in box list
	heat = add_heat(heat,hot_windows)
	    
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat,1)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)

	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(image), labels)
	return draw_img


# white_output = 'data/car_detection.mp4'
# clip1 = VideoFileClip("data/test_video.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)
import scipy.misc
images = glob.glob('data/video/*.jpg')

for image in images:
	img = mpimg.imread(image)
	draw_image = process_image(img)
	print(image)
	scipy.misc.imsave('data/images/'+image.split("/")[-1]+'.jpg', draw_image)
# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(draw_img)
# plt.title('Window')
# plt.subplot(122)
# plt.imshow(heatmap, cmap='hot')
# plt.title('HOG')
# fig.tight_layout()
# plt.savefig("data/heat.jpg")

