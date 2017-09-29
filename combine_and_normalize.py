import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
import time
import scipy
import pickle
from lesson_functions import *

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

###### TODO ###########
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for imgfile in imgs:
        # Read in each one by one
        img = mpimg.imread(imgfile)
        if img[0][0][0] > 0 and img[0][0][0] < 1 and 'png' in imgfile:
            img = (img * 255).astype('uint8')
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)
        # Apply bin_spatial() to get spatial color features
        feature_spatial = bin_spatial(feature_image, spatial_size)
        # Apply color_hist() to get color histogram features
        feature_hist = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((feature_spatial, feature_hist)))
    # Return list of feature vectors
    return features

images = glob.glob('input/vehicles/**/*.png')
non_car_images = glob.glob('input/non-vehicles/**/*.png')

cars = []
notcars = []
for image in images:
    cars.append(image)
for image in non_car_images:
    notcars.append(image)

print("car image count", len(cars))
print("non car image count", len(notcars))

spatial = 32
histbin = 32
        
t = time.time()
car_features = extract_features(cars, cspace='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))
notcar_features = extract_features(notcars, cspace='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))
print(round(time.time()-t, 2), 'Seconds to extract features')
timestamp = str(int(time.time()))
print("Extraction configurations: \n\t color space used = 'YCrCb'\n\t spacial size used is 32x32\n\thistogram range (0, 256)\n\t timestamp ", timestamp)

if len(car_features) > 0:

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # X = np.vstack((car_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars))
    # Plot an example of raw and scaled features

    # create the labels
    car_labels = np.ones(len(car_features))
    notcar_labels = np.zeros(len(notcar_features))

    y = np.hstack((car_labels, notcar_labels))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using spatial binning of:',spatial,
        'and', histbin,'histogram bins')

    pickle_file = "input/LinearSVC_histbin_32_spacial_32_"+timestamp+".pkl"
    # http://scikit-learn.org/stable/modules/model_persistence.html
    from sklearn.externals import joblib

    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')


    # print("svc data: ", pickle.dumps(svc))

    joblib.dump(svc, pickle_file)
    t3 = time.time()
    print(round(t3-t2, 2), 'Seconds to save SVC...')

    t3 = time.time()
    svc = joblib.load(pickle_file)
    t4 = time.time()
    print(round(t4-t3, 2), 'Seconds to load SVC...')
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

    print('My SVC predicts: ', svc.predict(X_test[0:10]))
    print('For labels: ', y_test[0:10])

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    # '''
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    # '''

    # with LinearSVC got 0.929898648649
    # Test Accuracy of SVC =  0.929898648649
    # Test Accuracy of SVC =  0.93384009009

    # fig = plt.figure(figsize=(12,4))
    # plt.subplot(131)
    # plt.imshow(mpimg.imread(cars[car_ind]))
    # plt.title('Original Image')
    # plt.subplot(132)
    # plt.plot(X[car_ind])
    # plt.title('Raw Features')
    # plt.subplot(133)
    # plt.plot(scaled_X[car_ind])
    # plt.title('Normalized Features')
    # fig.tight_layout()
    # plt.savefig("input/hog/test1_combine_and_normalize"+str(car_ind)+".jpg")
else: 
    print('Your function only returns empty feature vectors...')



# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


img = mpimg.imread("input/test1.jpg")
windows = slide_window(img, x_start_stop=[0, img.shape[1]], y_start_stop=[0,img.shape[0]//2], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
# print("windows", windows)

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    scipy.misc.imsave('input/images/'+str(int(time.time()))+'.jpg', imcopy)
    return imcopy

img = draw_boxes(img, windows)
fig = plt.figure(figsize=(12,4))
plt.subplot(131)
plt.imshow(img)
plt.title('Boxes on images')
plt.savefig("input/boxes.jpg")


'''
image count 8792
car image count 8792
non car image count 8968
33.32 Seconds to extract features
Using spatial binning of: 32 and 32 histogram bins
0.01 Seconds to load SVC...
Test Accuracy of SVC =  0.9659
My SVC predicts:  [ 1.  1.  0.  1.  0.  1.  0.  0.  0.  0.]
For these 10 labels:  [ 1.  1.  0.  1.  0.  1.  0.  1.  0.  0.]
0.00174 Seconds to predict 10 labels with SVC
'''




color_space = 'YCrCb'
spatial_size = (spatial, spatial)
hist_bins = histbin
# orient = 
# pix_per_cell = data['pix_per_cell']
# cell_per_block = data['cell_per_block']
# hog_channel = data['hog_channel']
# spatial_feat = data['spatial_feat']
# hist_feat = data['hist_feat']
# hog_feat = data['hog_feat']

# color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
# spatial_size = (16, 16) # Spatial binning dimensions
# hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = False # HOG features on or off
y_start_stop = [None, None]


image = mpimg.imread('input/test1.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255



def process_image(image):
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

    # for window in windows3:
    #     windows.append(window)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

    # draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)
    # draw_boxes(draw_image, windows1, color=(0, 0, 255), thick=6)
    # draw_boxes(draw_image, windows2, color=(0, 0, 255), thick=6)
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
    heat = apply_threshold(heat,1)

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

