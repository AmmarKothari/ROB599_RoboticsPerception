import logging
logging.basicConfig(level=logging.INFO)

import time
import numpy as np
import cv2
import pyrealsense as pyrs
from pyrealsense.constants import rs_option, rs_stream
import pdb


def nothing(x):
	pass


custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 30.0),
				  (rs_option.RS_OPTION_R200_LR_GAIN, 100.0)]

depth_fps = 30
depth_stream = pyrs.stream.DepthStream(fps=depth_fps)
dac_stream = pyrs.stream.DACStream(fps=depth_fps)
color_stream = pyrs.stream.ColorStream(fps=depth_fps)


cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('H_upper','image',0,255,nothing)
cv2.createTrackbar('H_lower','image',0,255,nothing)
cv2.createTrackbar('S_upper','image',0,255,nothing)
cv2.createTrackbar('S_lower','image',0,255,nothing)
cv2.createTrackbar('V_upper','image',0,255,nothing)
cv2.createTrackbar('V_lower','image',0,255,nothing)


# create switch for ON/OFF functionality
# switch = '0 : OFF \n1 : ON'
# cv2.createTrackbar(switch, 'image',0,1,nothing)


### NEED TO SWITCH TO RECTIFIED DEPTH IMAGE!!
with pyrs.Service() as serv:
	# with serv.Device(streams=(depth_stream,color_stream)) as dev:
	with serv.Device(streams=(depth_stream,color_stream, dac_stream)) as dev:

		dev.apply_ivcam_preset(0)

		try:  # set custom gain/exposure values to obtain good depth image
			custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 30.0),
							  (rs_option.RS_OPTION_R200_LR_GAIN, 100.0)]
			dev.set_device_options(*zip(*custom_options))
		except pyrs.RealsenseError:
			pass  # options are not available on all devices

		cnt = 0
		last = time.time()
		smoothing = 0.9
		fps_smooth = 30

		while True:

			cnt += 1
			if (cnt % 10) == 0:
				now = time.time()
				dt = now - last
				fps = 10/dt
				fps_smooth = (fps_smooth * smoothing) + (fps * (1.0-smoothing))
				last = now

			# check inputs
			# get current positions of four trackbars
			H_upper = cv2.getTrackbarPos('H_upper','image')
			H_lower = cv2.getTrackbarPos('H_lower','image')
			S_upper = cv2.getTrackbarPos('S_upper','image')
			S_lower = cv2.getTrackbarPos('S_lower','image')
			V_upper = cv2.getTrackbarPos('V_upper','image')
			V_lower = cv2.getTrackbarPos('V_lower','image')
			# s = cv2.getTrackbarPos(switch,'image')

			dev.wait_for_frames()
			pdb.set_trace()
			# # # # # # # # # # # # # # # # # # # # # #
			#		process color image
			# # # # # # # # # # # # # # # # # # # # # #
			c = dev.color
			c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)

			# remove background

			# use filters to only leave object pixels

			# use that mask to highlight depth image

			# get estimate from depth image
			# deproject_pixel_to_point(pixel, depth)

			# define range of blue color in HSV
			lower_blue = np.array([H_lower,S_lower,V_lower])
			upper_blue = np.array([H_upper,S_upper,V_upper])

			hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv, lower_blue, upper_blue)
			
			# Bitwise-AND mask and original image
			res = cv2.bitwise_and(c,c, mask= mask)


			# process depth image
			pdb.set_trace()
			d = dev.depth * dev.depth_scale * 1000
			d3 = np.stack((d,d,d), axis=2)
			d3 = (d3-np.min(d3))/(np.max(d3) - np.min(d3)) * 255
			d3 = d3.astype(np.uint8)

			# figuring out what the best way to extract information is




			# pdb.set_trace()
			# d2 = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_RAINBOW)
			d2 = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_HSV)
			# pdb.set_trace()

			cd = np.concatenate((res, d2, d3), axis=1)
			# cd = np.array(cd).tolist()
			cv2.putText(cd, str(fps_smooth)[:4], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))

			cv2.imshow('image', cd)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
