import cv2
import os
 


class recordImagesFromCamera(object):
 	def __init__(self, camera_port=0):
		# Camera 0 is the integrated web cam on my netbook
		self.camera_port = camera_port
		 
		#Number of frames to throw away while the camera adjusts to light levels
		self.ramp_frames = 30

		self.initCamera()
		self.setCameraParameters()

	def setCameraParameters(self, camera_width = 1280, camera_height = 720):
		self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
		self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
		self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

	# def __enter__(self):
	# 	#with Package() as package_obj:
	# 	# use package_obj
	# 	# https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
	# 	return self
		
	def initCamera(self):
		# Now we can initialize the camera capture object with the cv2.VideoCapture class.
		# All it needs is the index to a camera port.
		self.camera = cv2.VideoCapture()
		ret = self.camera.open(self.camera_port)
		if not ret:
			print("Could not open Camera")
			raise ValueError('Could not open Camera')
	 
	# Captures a single image from the camera and returns it in PIL format
	def _get_image(self):
		# read is the easiest way to get a full image out of a VideoCapture object.
		retval, im = self.camera.read()
		return im

	def get_image(self, file = "test_image.png"):
		self.rampCamera(self.ramp_frames)
		print("Taking image...")
		# Take the actual image we want to keep
		camera_capture = self._get_image()
		# A nice feature of the imwrite method is that it will automatically choose the
		# correct format based on the file extension you provide. Convenient!
		cv2.imwrite(file, camera_capture)

	def rampCamera(self, ramp_frames):
		# Ramp the camera - these frames will be discarded and are only used to allow v4l2
		# to adjust light levels, if necessary
		for i in xrange(ramp_frames):
			temp = self._get_image()
		
	def closeCamera(self):
		# You'll want to release the camera, otherwise you won't be able to create a new
		# capture object until your script exits
		del(camera)

	def userInput(self):
		# waits for user input to collect and save an image
		loop_flag = True
		i = 0
		while loop_flag:
			collect_flag = raw_input('Hit Enter to collect Data. Hit Q and then enter to collect data.')
			if collect_flag == '': # user hit enter
				fname = "Image_%s.jpg" %i
				self.get_image(fname)
				i +=1
			elif collect_flag.lower() == 'q':
				loop_flag = False



class recordVideoFromCamera(object):
	def __init__(self, fn, camera_port = 0):
		# subprocess.Popen(v4l2-ctl -d /dev/video1 --set-ctrl=focus_auto=0)
		self.camera_port = camera_port
		self.camera = cv2.VideoCapture(self.camera_port)
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.out_file = cv2.VideoWriter(fn,fourcc, 20.0, (640,480), True)


	def recordVideo(self):
		record_flag = False
		print('q to drop out of record \n r to (re)-start recording \n d to pause recording')
		while self.camera.isOpened():
			ret, frame = self.camera.read()
			if ret == True:
				if record_flag: self.out_file.write(frame)
				cv2.imshow('Frame', frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				elif cv2.waitKey(1) & 0xFF == ord('r'):
					print('Recording')
					record_flag = True
				elif cv2.waitKey(1) & 0xFF == ord('d'):
					record_flag = False
			else:
				print('Could not open camera')
				break
		self.out_file.release()
		cv2.destroyAllWindows()

	def closeCamera(self):
		self.camera.release()


class recordFrameFromVideo(object):
	# allows user to record a single frame from video file
	def __init__(self, fn):
		self.fn = fn
		self.loadFile(fn)

	def loadFile(self, fn):
		self.camera = cv2.VideoCapture(fn)

	def reloadFile(self):
		self.loadFile(self.fn)

	def saveFrame(self):
		frame_count = 1
		while self.camera.isOpened():
			ret, frame = self.camera.read()
			if ret == True:
				cv2.imshow('Frame', frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				elif cv2.waitKey(1) & 0xFF == ord('s'):
					cv2.imwrite('%04d_frame.png' %frame_count, frame)
					frame_count += 1
			else:
				print('Camera did not return frame')
				break
		cv2.destroyAllWindows()

	def saveAllFrames(self, folder):
		frame_count = 1
		while self.camera.isOpened():
			ret, frame = self.camera.read()
			if ret == True:
				cv2.imshow('Frame', frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				cv2.imwrite(os.path.join(folder, '%04d_frame.png' %frame_count), frame)
				frame_count += 1
			else:
				print('Camera did not return frame')
				break
		cv2.destroyAllWindows()






if __name__ == '__main__':
	R = recordImagesFromCamera()
	R.userInput()


	# VC = recordVideoFromCamera('background_subtraction.avi')