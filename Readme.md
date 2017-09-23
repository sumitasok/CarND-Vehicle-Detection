> Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
> Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
> Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.

 - png image read using matlib was scaled to 0-255
 - the data provided was split int 80 % training data and 20 % testing data
 - got an accuracy of 96-97 %

> Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

created boxes with 2 different widows and different overlaps.
![image](./data/boxes/boxes copy 3.jpg)

> Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
> Estimate a bounding box for vehicles detected.



[![IMAGE ALT TEXT](http://img.youtube.com/vi/hha0NsYXS5c/0.jpg)](http://www.youtube.com/watch?v= hha0NsYXS5c "Video Title")