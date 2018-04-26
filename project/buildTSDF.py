


'''
Classes to build a Truncated Signed distance function from depth camera
Expecting depth images from an intel RealSense R200.


'''
import numpy as np
import cv2
import pdb

class TSDF():
	def __init__(self, camera_intrinsics_fn):
		self.cam_int_fn = camera_intrinsics_fn

	def initializeConstants(self):
		self.voxel_grid_origin_x = -1.5 #Location of voxel grid origin in base frame camera coordinates
		self.voxel_grid_origin_y = -1.5
		self.voxel_grid_origin_z = 0.5
		# self.voxel_grid_origin_x = 0 #Location of voxel grid origin in base frame camera coordinates
		# self.voxel_grid_origin_y = 0
		# self.voxel_grid_origin_z = 0.5
		self.voxel_size = 0.006
		self.voxel_grid_dim_x = 500
		self.voxel_grid_dim_y = 500
		self.voxel_grid_dim_z = 500
		self.trunc_margin = self.voxel_size * 5
		# self.trunc_margin = self.voxel_size * 5





	def printFN(self, fn):
		print("File: {}".format(fn))

	def loadCameraIntrinsics(self, fn=None):
		if fn is None:
			fn = self.cam_int_fn
		self.cam_K = np.zeros((3,3))
		with open(fn, 'rb') as f:
			i = 0
			for ln in f.readlines():
				self.cam_K[i,:] = np.array(ln.replace('\t', '').replace('\n','').split()).astype('float')
				i += 1
		return self.cam_K

	def loadDepthImage(self, fn):
		return cv2.imread(fn)

	def loadPose(self, fn):
		i =1

	def initializeTSDFGrid(self, dims):
		# dims is the number of grid points
		# how to correllate to physical space?
		self.dims = dims
		self.voxel_grid_TSDF = np.ones(dims).astype('float')
		self.voxel_grid_weight = np.ones(dims).astype('float')

	# def rotatePointsIntoWorld(self, base_pose, camera_pose):
		# don't need to do this yet because all the points are from the same perspective
		# probably need this to get physical meaning from points
		# multiply camera_pose with base_pose

	def Integrate(self, depth_img):
		# need to loop through all three dimensions
		im_width = depth_img.shape[1]
		im_height = depth_img.shape[0]
		for pt_grid_x in range(self.dims[0]):
			for pt_grid_y in range(self.dims[1]):
				for pt_grid_z in range(self.dims[2]):
					# Convert voxel center from grid coordinates to base frame camera coordinates
					pt_base_x = self.voxel_grid_origin_x + pt_grid_x * self.voxel_size
					pt_base_y = self.voxel_grid_origin_y + pt_grid_y * self.voxel_size
					pt_base_z = self.voxel_grid_origin_z + pt_grid_z * self.voxel_size

					# Convert from base frame camera coordinates to current frame camera coordinates
					tmp_pt = np.zeros(3);
					# apply translation
					# tmp_pt[0] = pt_base_x - cam2base[0, 3];
					# tmp_pt[1] = pt_base_y - cam2base[1, 3];
					# tmp_pt[2] = pt_base_z - cam2base[2, 3];
					# # apply rotation (do this with just a rotation)
					# pt_cam_x = cam2base[0, 0] * tmp_pt[0] + cam2base[1, 0] * tmp_pt[1] + cam2base[2, 0] * tmp_pt[2];
					# pt_cam_y = cam2base[0, 1] * tmp_pt[0] + cam2base[1, 1] * tmp_pt[1] + cam2base[2, 1] * tmp_pt[2];
					# pt_cam_z = cam2base[0, 2] * tmp_pt[0] + cam2base[1, 2] * tmp_pt[1] + cam2base[2, 2] * tmp_pt[2];

					pt_cam_x, pt_cam_y, pt_cam_z = pt_base_x, pt_base_y, pt_base_z



					 # if its a negative distance than ignore it
					if (pt_cam_z <= 0):
					  continue;

					# create an image of SDF from camera point of view
					# reject if it is outside camera view
					pt_pix_x = round(self.cam_K[0, 0] * (pt_cam_x / pt_cam_z) + self.cam_K[0, 2])
					pt_pix_y = round(self.cam_K[1, 1] * (pt_cam_y / pt_cam_z) + self.cam_K[1, 2])
					if (pt_pix_x < 0 or pt_pix_x >= im_width or pt_pix_y < 0 or pt_pix_y >= im_height):
						continue;

					# get value from depth image
					# which chanell to use from depth image?
					# what are the units of these values?
					depth_val = depth_img[np.round(pt_pix_y).astype('int'), np.round(pt_pix_x).astype('int'), 0];
					
					# reject if its too close or too far
					if depth_val <= 0 or depth_val >= 6:
						continue
					# pdb.set_trace()
					# difference between current estimate and measurement
					diff = depth_val - pt_cam_z;
					# don't update if measured point is way beyond the location of a surface -- might be rejecting noise
					if (diff <= -self.trunc_margin):
						continue
					
					volume_idx = (pt_grid_x, pt_grid_y, pt_grid_z);
					dist = min(1.0, diff / self.trunc_margin) # what is this?
					# dist = diff / self.trunc_margin
					print("X pixel: {}, Y Pixel: {}, Depth: {}, Truncation: {}".format(pt_pix_x, pt_pix_y, depth_val, dist))
					weight_old = self.voxel_grid_weight[volume_idx]
					weight_new = weight_old + 1.0
					self.voxel_grid_weight[volume_idx] = weight_new;
					self.voxel_grid_TSDF[volume_idx] = (self.voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
		# pdb.set_trace()
		# self.SDFtoPLY('test.ply', self.voxel_grid_TSDF, self.voxel_grid_TSDF)

	def SDFtoPLY(self, fn, SDF, SDF_weights):
		# convert from SDF to a point cloud
		num_pts = 0

		points_string = ''
		tsdf_thresh = 0.2 # from the example
		weight_thresh = 0.0
		SDF_dims = SDF.shape
		for i_x in range(SDF_dims[0]):
			for i_y in range(SDF_dims[1]):
				for i_z in range(SDF_dims[2]):
					# skip over if it didn't get enough measurements or if it was too small
					if abs(SDF[i_x, i_y, i_z]) < tsdf_thresh and SDF_weights[i_x, i_y, i_z] > weight_thresh:
						# calculate total number of points
						num_pts += 1
						pt_base_x = self.voxel_grid_origin_x + i_x*self.voxel_size
						pt_base_y = self.voxel_grid_origin_y + i_y*self.voxel_size
						pt_base_z = self.voxel_grid_origin_z + i_z*self.voxel_size
						# print("{}, {}, {}".format(pt_base_x, pt_base_y, pt_base_z))
						# pdb.set_trace()
						points_string += str(pt_base_x) + " "
						points_string += str(pt_base_y) + " "
						points_string += str(pt_base_z) + " "
		pdb.set_trace()



		with open(fn, 'wb') as f:
			f.write("ply\n")
			f.write("format binary_little_endian 1.0\n")
			f.write("element vertex %d\n" %num_pts)
			f.write("property float x\n")
			f.write("property float y\n")
			f.write("property float z\n")
			f.write("end_header\n")
			f.write(points_string)




	# function to resample grid size

	# function to resample grid according to oct-tree

	# def updateTSDF(self, depth_img, pose):
		# convert depth image into base frame

		# integrate poits






