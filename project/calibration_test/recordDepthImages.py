# import rospy
# import rosbag
# import sensor_msgs.msg
import pdb
import cv2
import os
import time
# import cv_bridge # CvBridge, CvBridgeError

import pyrealsense as pyrs
from pyrealsense.constants import rs_option
import numpy as np


depth_fps = 60
depth_stream = pyrs.stream.DepthStream(fps=depth_fps)
color_stream = pyrs.stream.ColorStream(fps=depth_fps)

topics = ['/camera/rgb/image_raw', '/camera/depth/image_rect_raw']

class recordDepthImages():
	def __init__(self, fn):
		self.bag = rosbag.bag.Bag(fn, mode='w', compression='lz4')

	def startNode(self):
		rospy.init_node('DepthRecorder', anonymous=True)

	def record(self):
		data = self.getMessages(topics)
		self.writeToBag(data)

	def getMessages(self, topics):
		d = dict.fromkeys(topics)
		for t in topics:
			d[t] = rospy.wait_for_message(t, sensor_msgs.msg.Image)
		return d

	def writeToBag(self, data):
		for t in topics:
			self.bag.write(t, data[t])

	def shutdown(self):
		self.bag.close()

class extractImagesFromBag():
	def __init__(self, fn):
		self.bagFileNameCheck(fn)
		self.img_count = 0

	def openBag(self):
		self.bag = self.loadBag()
		self.bag_gen = self.bag.read_messages()


	def loadBag(self): # load bag file into object
		# does some error correction
		try:
			b = rosbag.Bag(self.fn, "r")
		except rosbag.bag.ROSBagUnindexedException:
			print("unindexed bag... Trying to reindex.")
			p = subprocess.Popen(['rosbag', 'reindex', self.fn])
			try:
				b = rosbag.Bag(self.fn, "r")
			except:
				print("Could not reindex and open ")
				raise IOError
		print("Bag File Loaded: %s" %self.fn)
		return b

	def bagFileNameCheck(self, fn): # check file name
		if '.bag' not in fn:
			fn += '.bag' # add .bag if it isn't in the file name?  I suppose you could have a file that doesn't end in bag but that seems confusing
		self.fn = fn

	def iterateThroughBag(self, folderName='.'):
		# assumes sequential order of pairs
		bridge = cv_bridge.CvBridge()
		for topic, msg, t in self.bag_gen:
			cv_image = bridge.imgmsg_to_cv2(msg)
			i = self.img_count/2
			if 'rgb' in topic:
				img_t = 'color'
			else:
				img_t = 'depth'
			fn_name = os.path.join(folderName, 'frame-%04d.%s.png' %(i,img_t))
			self.img_count += 1
			cv2.imwrite(fn_name, cv_image)


class recordDepthRealsense(object):
	# record depth through the pyrealsense package
	def __init__(self):
		self.serv = pyrs.Service()
		self.dev = self.serv.Device(streams=(depth_stream,color_stream))
		self.dev.apply_ivcam_preset(0)

		try:  # set custom gain/exposure values to obtain good depth image
			custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 30.0),
							  (rs_option.RS_OPTION_R200_LR_GAIN, 100.0)]
			self.dev.set_device_options(*zip(*custom_options))
		except pyrs.RealsenseError:
			pass  # options are not available on all devices
		time.sleep(1)

	def getImages(self):
		self.dev.wait_for_frames()

	def recordImages(self, fn_depth, fn_depth2, fn_rgb):
		d = self.recordDepth(fn_depth, fn_depth2)
		c = self.recordRGB(fn_rgb)
		cv2.imshow('Depth', d)
		cv2.imshow('Color', d)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			return 0
		else:
			return 1

	def recordDepth(self, fn, fn2):
		d = self.dev.depth
		d_color = self.convert_z16_to_bgr(d)
		cv2.imwrite(fn, d)
		# fn2 = fn.split('.')
		# fn2.insert(-1,'_color')
		# fn2 = '.'.join(fn2)
		cv2.imwrite(fn2, d_color)
		return d

	def recordRGB(self, fn):
		c = self.dev.color
		c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
		cv2.imwrite(fn, c)
		return c

	def convert_z16_to_bgr(self, frame):
		'''Performs depth histogram normalization

		This raw Python implementation is slow. See here for a fast implementation using Cython:
		https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/cython_methods/methods.pyx
		'''
		hist = np.histogram(frame, bins=0x10000)[0]
		hist = np.cumsum(hist)
		hist -= hist[0]
		rgb_frame = np.empty(frame.shape[:2] + (3,), dtype=np.uint8)

		zeros = frame == 0
		non_zeros = frame != 0

		f = hist[frame[non_zeros]] * 255 / hist[0xFFFF]
		rgb_frame[non_zeros, 0] = 255 - f
		rgb_frame[non_zeros, 1] = 0
		rgb_frame[non_zeros, 2] = f
		rgb_frame[zeros, 0] = 20
		rgb_frame[zeros, 1] = 5
		rgb_frame[zeros, 2] = 0
		return rgb_frame






if __name__ == '__main__':
	# r = recordDepthImages('Depth_Object.bag')
	# r.startNode()
	# rate = rospy.Rate(10)


	# while not rospy.is_shutdown():
	# 	r.record()
	# 	try:
	# 		rate.sleep()
	# 	except rospy.ROSInterruptException:
	# 		r.shutdown()
	# rospy.on_shutdown(r.shutdown)

	# print('Shutting Down!')
	# pdb.set_trace()

	r = recordDepthRealsense()
	i = 0
	while True:
		r.getImages()
		folder =  'RGBDTestScene/Attempt2'
		r.recordImages(os.path.join(folder, '%4d_depth.png' %i), os.path.join(folder, '%4d_color_depth.png' %i), os.path.join(folder, '%4d_rgb.png' %i))
		i = i+1
