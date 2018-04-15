import cv2
 


class recordImagesFromCamera(object):
 	def __init__(self, camera_port=0):
		# Camera 0 is the integrated web cam on my netbook
		self.camera_port = camera_port
		 
		#Number of frames to throw away while the camera adjusts to light levels
		self.ramp_frames = 30

		self.initCamera()

	# def __enter__(self):
	# 	#with Package() as package_obj:
	# 	# use package_obj
	# 	# https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
	# 	return self
		

	def initCamera(self):
		# Now we can initialize the camera capture object with the cv2.VideoCapture class.
		# All it needs is the index to a camera port.
		self.camera = cv2.VideoCapture(self.camera_port)
	 
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



if __name__ == '__main__':
	R = recordImagesFromCamera()
	R.userInput()