
import pdb, os
import numpy as np
import cv2



class PreProcessVideo(object):
	def __init__(self, fn, out_folder):
		i = 1
		self.fn = fn
		self.out_folder = os.path.join(out_folder, 'images') #has to be in a folder called images
		if not os.path.exists(out_folder):
			os.makedirs(out_folder)

	def seperateImages(self, limit = None):
		vidcap = cv2.VideoCapture(self.fn)
		success,image = vidcap.read()
		count = 0
		success = True
		y, x, d = image.shape
		h = y/4
		w = x/4
		while success:
			success,image = vidcap.read()
			print('Read a new frame: ', success)
			# cv2.imwrite(os.path.join(self.out_folder, "frame%d.jpg" % count), image)     # save frame as JPEG file
			image_crop = self.cropImage(image, y/2-h, x/2-w, h, w)
			cv2.imwrite(os.path.join(self.out_folder, "frame%d.jpg" % count), image_crop)     # save frame as JPEG file
			count += 1
			if limit is not None and count > limit:
				success = False

	def cropImage(self, image, y_start, x_start, h, w):
		crop_img = image[y_start:y_start+h, x_start:x_start+w]
		# cv2.imshow("cropped", crop_img)
		# cv2.waitKey(0)
		return crop_img

if __name__ == '__main__':
	P = PreProcessVideo('IMG_1546.MOV', 'SeperatedImages')
	P.seperateImages()