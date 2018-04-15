import numpy as np
import os
import pdb
# i did not record "poses" from the realsense during runtime
# so i am making them up now


fakePose = np.eye(4)

class makeFakePoses():
	def __init__(self):
		i = 1

	def getDepthFiles(self, p='.'):
		return [os.path.join(p,fn) for fn in os.listdir(p) if 'depth' in fn]

	def generatePoseFileNames(self, depthFNs):
		return [fn.replace('depth.png', 'pose.txt') for fn in depthFNs]
		

	def createPoses(self, pose, fns):
		for fn in fns:
			with open(fn, 'wb') as f:
				for ps in pose:
					f.write('{:8E} \t {:8E} \t {:8E} \t {:8E} \n'.format(*ps).expandtabs(4))

					





if __name__ == '__main__':
	m = makeFakePoses()
	dfns = m.getDepthFiles('WithObject')
	poseFns = m.generatePoseFileNames(dfns)
	m.createPoses(fakePose, poseFns)
