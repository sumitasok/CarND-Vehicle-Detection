import cv2
vidcap = cv2.VideoCapture('data/test_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  cv2.imwrite("data/video/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
  
# http://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames