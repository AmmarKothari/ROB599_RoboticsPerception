import numpy as np


def horizConcat(img_list):
	img1 = img_list[0]
	for i in range(1, len(img_list)):
		img2 = img_list[i]
		# dealing with gray scale
		if len(img1.shape) == 2:
			img1 = np.stack((img1, img1, img1), axis=2)
		if len(img2.shape) == 2:
			img2 = np.stack((img2, img2, img2), axis=2)
		y = min(img1.shape[0], img2.shape[0])
		x = min(img1.shape[1], img2.shape[1])
		if len(img1.shape) == 3 and len(img2.shape) == 3:
			c = min(img1.shape[2], img2.shape[2])
			vis = np.concatenate((img1[:y,:,:c], img2[:y,:,:c]), axis=1)
		img1 = vis
	return vis

def addChannels(img):
	img1 = np.stack((img, img, img), axis=2)
	return img1


def avgImage(img1, img2):
	y = min(img1.shape[0], img2.shape[0])
	x = min(img1.shape[1], img2.shape[1])
	c = min(img1.shape[2], img2.shape[2])
	img_avg = 0.5*(img1[:y,:x,:c]+img2[:y,:x,:c])
	return img_avg