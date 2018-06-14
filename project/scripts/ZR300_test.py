

import cv2, os, pdb
import numpy as np
## setup logging
import logging
logging.basicConfig(level = logging.INFO)

## import the package
import pyrealsense as pyrs

from utils import addChannels



fourcc = cv2.VideoWriter_fourcc(*'XVID')
camera_color = cv2.VideoWriter('DepthTESTVIDEO_color.avi',fourcc, 20.0, (640,480), True)
camera_depth = cv2.VideoWriter('DepthTESTVIDEO_depth.avi',fourcc, 20.0, (640,480), True)
camera_cad = cv2.VideoWriter('DepthTESTVIDEO_cad.avi',fourcc, 20.0, (640,480), True)


font = cv2.FONT_HERSHEY_SIMPLEX



SAVEFOLDER = 'DepthTest_ZR300'
FPS = 60


## start the service - also available as context manager
serv = pyrs.Service()
# Serial Number: 3481802454


## create a device from device id and streams of interest
# cam = serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps = FPS), pyrs.stream.DepthStream(fps = FPS)])
# cam = serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps = FPS), pyrs.stream.DepthStream(fps = FPS), pyrs.stream.DACStream(fps = FPS)])
cam = serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps = FPS), pyrs.stream.DepthStream(fps = FPS), pyrs.stream.CADStream(fps = FPS)])


## retrieve 60 frames of data
i = 0
while True:
	cam.wait_for_frames()
	fn_color = os.path.join(SAVEFOLDER, 'color_{}.jpg'.format(i))
	cv2.imwrite(fn_color, cam.color)
	fn_depth = os.path.join(SAVEFOLDER, 'depth_{}.jpg'.format(i))
	fn_depth_np = os.path.join(SAVEFOLDER, 'depth_{}'.format(i))
	np.save(fn_depth_np, cam.depth)
	cv2.imwrite(fn_depth, cam.depth)
	# cv2.imshow('', cam.dac)

	camera_color.write(cam.color)
	depth_out = np.uint8(np.array(cam.depth*255).astype(int))
	camera_depth.write(addChannels(depth_out))
	camera_cad.write(cam.cad)

	cv2.imshow('', cam.cad)
	if cv2.waitKey(100) == ord('q'):
		break
	i += 1

camera_color.release()
camera_depth.release()
camera_cad.release()

## stop camera and service
cam.stop()
serv.stop()