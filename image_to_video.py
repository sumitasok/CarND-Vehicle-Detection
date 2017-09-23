import cv2
import glob

images = glob.glob('data/images/*.jpg')

img1 = cv2.imread(images[0])
height , width , layers =  img1.shape


fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
video = cv2.VideoWriter("video.mp4", fourcc, 20.0, (width, height))

# video = cv2.VideoWriter('data/video.avi',-1,1,(width,height))

for image in images:
	print(image)
	img1 = cv2.imread(image)
	video.write(img1)

cv2.destroyAllWindows()
video.release()

# http://stackoverflow.com/questions/14440400/creating-a-video-using-opencv-2-4-0-in-python