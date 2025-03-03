'''
file for testing out parts of the TSDF class
'''




import glob
import cv2
import os
import numpy as np
from buildTSDF import TSDF

pic_dir = '/home/ammar/Documents/Projects/ROB599_RoboticsPerception/project/calibration_test/RGBDTestScene/WithoutObject'
pose_fns = glob.glob(os.path.join(pic_dir,'*.txt'))
depth_fns = glob.glob(os.path.join(pic_dir, '*.depth.png'))


grid_dims = (500,500,500) # x,y,z

TSDF = TSDF('test.txt')
TSDF.initializeConstants()
TSDF.initializeTSDFGrid(grid_dims)
TSDF.loadCameraIntrinsics(os.path.join(pic_dir, 'camera-intrinsics.txt'))
c = 10
for depth_fn, pose_fn in zip(depth_fns[0:c], pose_fns)[0:c]:
	img = TSDF.loadDepthImage(depth_fns[0])
	pose = TSDF.loadPose(pose_fns[0])
	TSDF.Integrate(img)


TSDF.SDFtoPLY('test.ply', TSDF.voxel_grid_TSDF, TSDF.voxel_grid_TSDF)







