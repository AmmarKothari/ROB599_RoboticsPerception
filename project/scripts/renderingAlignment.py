


import meshrender
import trimesh
import autolab_core
import perception
import pyquaternion

import cv2
import pdb
import os
import copy

from utils import horizConcat, avgImage




camera = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_file = cv2.VideoWriter('AlignmentPresentation.avi',fourcc, 20.0, (640*2,480), True)
font = cv2.FONT_HERSHEY_SIMPLEX



SAVEFOLDER = "DepthAlignment"
EXAMPLE = os.path.join(SAVEFOLDER, "ExampleDepth1.png")

### Make into a class so don't have to keep doing setup

# load the object
coke_mesh = trimesh.load_mesh('coke.obj')

# set some transformation -- at the origin, no rotation
# autolab_core.RigidTransform.random_rotation()

rot_start = np.array([[-0.56806586, -0.10094891, -0.81676832],
						[-0.56244744, -0.67688651,  0.47484474],
						[-0.60079452,  0.72913234,  0.32773766]])
coke_pose = autolab_core.RigidTransform(
    rotation=np.eye(3),
    # rotation=rot_start,
    translation=np.array([0.0, 0.0, 0.0]),
    from_frame='obj',
    to_frame='world'
)

# set object properties -- don't need this i think if only doing depth
coke_material = meshrender.MaterialProperties(
    # color = 5.0*np.array([0.1, 0.1, 0.1]),
    k_a = 0.3,
    k_d = 0.5,
    k_s = 0.2,
    alpha = 10.0,
    smooth=False,
    wireframe=False
)

# Create SceneObjects for object
coke_obj = meshrender.SceneObject(coke_mesh, coke_pose, coke_material)

# what does this line do
# coke_inst_obj = meshrender.InstancedSceneObject(coke_mesh, [coke_pose], colors=np.array([[0,0,1],[0,1,0]]), material=coke_material)

# Initialize Scene
scene = meshrender.Scene()
scene.add_object('coke_can', coke_obj)

# Add lighting -- probably don't need this for depth
# Create an ambient light
ambient = meshrender.AmbientLight(
    color=np.array([1.0, 1.0, 1.0]),
    strength=1.0
)

dl = meshrender.DirectionalLight(
    direction=np.array([0.0, 0.0, -1.0]),
    color=np.array([1.0, 1.0, 1.0]),
    strength=2.0
)

# Add the lights to the scene
scene.ambient_light = ambient # only one ambient light per scene
scene.add_light('direc', dl)

#====================================
# Add a camera to the scene
#====================================

# Set up camera intrinsics
ci = perception.CameraIntrinsics(
    frame = 'camera',
    fx = 525.0,
    fy = 525.0,
    cx = 320.0,
    cy = 240.0,
    skew=0.0,
    height=480,
    width=640
)

# Set up the camera pose (z axis faces away from scene, x to right, y up)
cp = autolab_core.RigidTransform(
    rotation = np.array([
        [0.0, 0.0, 1.0],
        [0.0, -1.0,  0.0],
        [1.0, 0.0,  0.0]
    ]),
    translation = np.array([-10, 0.0, 0.0]),
    from_frame='camera',
    to_frame='world'
)


# obj_guess_q = pyquaternion.Quaternion(axis=[1,0,0], radians=0)
# pdb.set_trace()

# Create a VirtualCamera
camera = meshrender.VirtualCamera(ci, cp)

# Add the camera to the scene
scene.camera = camera
color_image_raw, depth_image_raw = scene.render(render_color=True) # call it once to load rendere

## given a test image
test_depth = cv2.imread(EXAMPLE)


# pdb.set_trace()




i = 0
epsilon = 200 # pixelwise depth difference for the same pixel

best_q = coke_pose.quaternion
best_Tl = coke_pose.translation

color_image_raw, depth_image_raw = scene.render(render_color=True)
best_score = cv2.norm(depth_image_raw-test_depth[:,:,0])
last_score = best_score

iter_count = 0
conv_count = 0
CONV_LIMIT = 5

delta_score = 1e-2

while best_score > epsilon:
	old_q = best_q
	old_Tl = best_Tl
	old_score = best_score

	all_qs = []
	all_Tls = []
	all_scores = []

	all_qs.append(best_q)
	all_Tls.append(best_Tl)
	all_scores.append(best_score)

	temp = np.exp(-iter_count/50)
	# temp = abs(best_score - epsilon)/2000
	for i in range(10):
		# generate new object in new position
		new_Tl = old_Tl
		new_q = old_q
		if np.remainder(iter_count,2) == 0: # update quat
			new_q = old_q + np.random.randn(4)*temp
			new_q = new_q / np.linalg.norm(new_q)
			# pdb.set_trace()
			# print('Quaternion: {}'.format(new_q))
		else:
			new_Tl = old_Tl + (np.random.randn(3) *temp)

		new_R = autolab_core.RigidTransform.rotation_from_quaternion(new_q)

		coke_pose = autolab_core.RigidTransform(
		    rotation=new_R,
		    translation=new_Tl,
		    from_frame='obj',
		    to_frame='world'
		)

		coke_obj = meshrender.SceneObject(coke_mesh, coke_pose, coke_material)
		
		# remove old object from scene
		scene.remove_object('coke_can')

		# add new object to scene
		scene.add_object('coke_can', coke_obj)

		# render
		# rotate the object in the scene

		# pdb.set_trace()
		color_image_raw, depth_image_raw = scene.render(render_color=True)

		# evaluate score
		score = cv2.norm(depth_image_raw-test_depth[:,:,0])
		
		all_qs.append(new_q)
		all_Tls.append(new_Tl)
		all_scores.append(score)

		# depth_image_raw = scene.render(render_color=False)
		cv2.imwrite(os.path.join(SAVEFOLDER, "color_{}_{}.png".format(iter_count, i)), color_image_raw)
		cv2.imwrite(os.path.join(SAVEFOLDER, "depth_{}_{}.png".format(iter_count, i)), depth_image_raw)


		
		vis = horizConcat((test_depth, depth_image_raw))

		# ret,thresh1 = cv2.threshold(vis[:,:,0],50,255,cv2.THRESH_BINARY)
		# pdb.set_trace()
		# vis = cv2.bitwise_and(vis, vis, mask=thresh1)
		# vis[vis>50] = 255
		vis_out = np.uint8(np.array(vis*255).astype(int))
		cv2.putText(vis_out,'Best Score: {}'.format(best_score),(0,400), font, 1,(255,255,255),2,cv2.LINE_AA)
		cv2.putText(vis_out,'X:{}, Y:{}, Z:{}'.format(*best_Tl),(0,430), font, 1,(255,255,255),2,cv2.LINE_AA)
		cv2.putText(vis_out,'Q0:{}, Q1:{}, Q2:{}, Q3:{}'.format(*best_q),(0,460), font, 1,(255,255,255),2,cv2.LINE_AA)

		cv2.imshow('', vis_out)
		out_file.write(vis_out)
		# cv2.normalize(vis, vis, 0, 1, cv2.NORM_MINMAX)
		# out_file.write(np.uint8(np.array(vis*255).astype(int)))
		cv2.waitKey(100)

	# update best guess so far
	best_ind = np.argmin(all_scores)
	print("Best Score: {}".format(all_scores[best_ind]))
	best_q = copy.deepcopy(all_qs[best_ind])
	best_Tl = copy.deepcopy(all_Tls[best_ind])
	best_score = copy.deepcopy(all_scores[best_ind])
	# pdb.set_trace()
	iter_count += 1

	if last_score == best_score:
		conv_count += 1
	else:
		conv_count = 0

	if conv_count >= CONV_LIMIT:
		break

	last_score = best_score

for i in range(100): # just add a few frames at the end
	out_file.write(vis_out)





# v = meshrender.SceneViewer(scene, raymond_lighting=True)



# rand_R = autolab_core.RigidTransform.random_rotation()
# rand_TL = np.random.rand(3,1)
# rand_T = np.vstack((np.hstack((rand_R, rand_TL)), [0,0,0,1]))

# while distance is greater than some epsilon

## rotate the object

## take an image

# calculate distance

