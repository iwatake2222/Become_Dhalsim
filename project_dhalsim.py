import numpy as np
import scipy.ndimage
import scipy.misc
import cv2
from bodypix.pose_engine import PoseEngine, EDGES, BODYPIX_PARTS

# Consider left arm on the screen is LEFT(not the real left arm)
RIGHT_ARM_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if  "left lower arm" in v or "left hand" in v]
LEFT_ARM_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "right lower arm" in v or "right hand" in v]

def clip_heatmap(heatmap, v0, v1):
	a = v0 / (v0 - v1);
	b = 1.0 / (v1 - v0);
	return np.clip(a + b * heatmap, 0.0, 1.0);

class BodyPart:
	def __init__(self, heatmap, bodypart, image_original):
		self.image_original = image_original

		# (h,w) the same size as heatmap e.g. (31,41)
		# 0: the point is not categorized in the part, 1: the point is categorized in the part
		self.mask = heatmap * (np.sum(bodyparts[:,:,bodypart], axis=2) - 0.5) * 100
		self.mask = 1 * np.clip(self.mask, 0, 1)
		return

	def get_mask_with_resize(self):
		rescale_factor = [
			self.image_original.shape[0] / self.mask.shape[0],
			self.image_original.shape[1] / self.mask.shape[1]
		]
		return scipy.ndimage.zoom(self.mask, rescale_factor, order=0)
	
	def crop_body_part(self, dsize=None, fx=1, fy=1):
		mask_in_original = self.get_mask_with_resize()
		pos_y, pos_x = np.where(mask_in_original != 0)
		image_masked = self.image_original * np.dstack([mask_in_original] * 3)
		image_masked = np.uint8(np.clip(image_masked, 0, 255))
		image_cropped = image_masked[np.min(pos_y) : np.max(pos_y), np.min(pos_x) : np.max(pos_x),]
		image_cropped = cv2.resize(image_cropped, dsize=dsize, fx=fx, fy=fy)
		return image_cropped

	def crop_mask_image_with_resize(self, dsize=None, fx=1, fy=1):
		mask_in_original = self.get_mask_with_resize()
		pos_y, pos_x = np.where(mask_in_original != 0)
		mask_in_original= np.dstack([255 * (1 - mask_in_original)] * 3)
		mask_in_original = np.uint8(np.clip(mask_in_original, 0, 255))
		mask_cropped = mask_in_original[np.min(pos_y) : np.max(pos_y), np.min(pos_x) : np.max(pos_x),]
		mask_cropped = cv2.resize(mask_cropped, dsize=dsize, fx=fx, fy=fy)
		return mask_cropped

	def get_pos_in_original(self):
		mask_in_original = self.get_mask_with_resize()
		pos_y, pos_x = np.where(mask_in_original != 0)
		return np.min(pos_x), np.max(pos_x), np.min(pos_y), np.max(pos_y)

	def judge_part_is_in_left(self, nose_x):
		pos_y, pos_x = np.where(self.mask != 0)
		if(len(pos_x) > 0):
			distance_left = nose_x - np.min(pos_x)
			distance_right = np.max(pos_x) - nose_x
			if distance_left > distance_right:
				return True
			return False
		# shouldn't reach here
		return False

	def judge_part_is_straight(self):
		pos_y, pos_x = np.where(self.mask != 0)
		if(len(pos_x) > 0):
			most_left = (pos_x[np.argmin(pos_x)], pos_y[np.argmin(pos_x)])
			most_right = (pos_x[np.argmax(pos_x)], pos_y[np.argmax(pos_x)])
			left_arm_width = np.max(pos_x) - np.min(pos_x)
			left_arm_height = np.max(pos_y) - np.min(pos_y)
			if left_arm_height == 0: left_arm_height = 1
			if left_arm_width > 4 and left_arm_width / left_arm_height > 2 and abs(most_left[1] - most_right[1]) < 4:
				return True
		return False

def mask_image(image_org, mask):
	image_org = image_org * np.dstack([1 - mask] * 3)
	image_org = np.uint8(np.clip(image_org, 0, 255))
	return image_org


def stretchRight(image, body_part):
	mask_pos = body_part.get_pos_in_original()
	max_width = image.shape[1] - mask_pos[0]
	fx = min(3, max_width / (mask_pos[1] - mask_pos[0]))
	mask_image = body_part.crop_mask_image_with_resize(fx=fx)
	part_image = body_part.crop_body_part(fx=fx)
	image[mask_pos[2] : mask_pos[3], mask_pos[0] : mask_pos[0] + mask_image.shape[1]] &= mask_image
	image[mask_pos[2] : mask_pos[3], mask_pos[0] : mask_pos[0] + mask_image.shape[1]] |= part_image
	return

def stretchLeft(image, body_part):
	mask_pos = body_part.get_pos_in_original()
	max_width = mask_pos[1]
	fx = min(3, max_width / (mask_pos[1] - mask_pos[0]))
	mask_image = body_part.crop_mask_image_with_resize(fx=fx)
	part_image = body_part.crop_body_part(fx=fx)
	image[mask_pos[2] : mask_pos[3], mask_pos[1] - mask_image.shape[1] : mask_pos[1]] &= mask_image
	image[mask_pos[2] : mask_pos[3], mask_pos[1] - mask_image.shape[1] : mask_pos[1]] |= part_image
	return

def get_nose_position(poses):
	nose_pos = (0, 0)
	for pose in poses:
		for label, keypoint in pose.keypoints.items():
			if keypoint.score < 0.3: continue
			if label == "nose":
				nose_pos = ((int(keypoint.yx[0]), int(keypoint.yx[1])))
	return nose_pos


# Initialize
model = 'bodypix/models/bodypix_mobilenet_v1_075_640_480_16_quant_edgetpu_decoder.tflite'
engine = PoseEngine(model)
inference_size = (engine.image_width, engine.image_height)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cnt_stretch_l = 0
cnt_stretch_r = 0

# Main loop
while True:
	# Capture image
	# key = cv2.waitKey(-11)
	# image_org = cv2.imread("human.jpg")
	ret, image_org = cap.read()
	image_display = image_org.copy()
	image = cv2.resize(image_org, inference_size)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.astype(np.uint8)

	# Run inference
	inference_time, poses, heatmap, bodyparts = engine.DetectPosesInImage(image)

	# Post-process
	## Find nose position to judge if arm is on left or right
	nose_pos = get_nose_position(poses)
	nose_x_in_heatmap = nose_pos[1] * heatmap.shape[1] / engine.image_width

	## Retrieve heatmap of left/right arms
	heatmap = clip_heatmap(heatmap, -1.0, 1.0)
	l_arm = BodyPart(heatmap, LEFT_ARM_BODYPARTS, image_org)
	r_arm = BodyPart(heatmap, RIGHT_ARM_BODYPARTS, image_org)
	
	## Stretch arm if the arm is straight horizontally for a while (more than 3 frames)
	cnt_stretch_l = cnt_stretch_l +1 if l_arm.judge_part_is_straight() == True else 0
	cnt_stretch_r = cnt_stretch_r +1 if r_arm.judge_part_is_straight() == True else 0

	if cnt_stretch_l > 3:
		if l_arm.judge_part_is_in_left(nose_x_in_heatmap) == True:
			stretchLeft(image_display, l_arm)
		else:
			stretchRight(image_display, l_arm)
	if cnt_stretch_r > 3:
		if r_arm.judge_part_is_in_left(nose_x_in_heatmap) == True:
			stretchLeft(image_display, r_arm)
		else:
			stretchRight(image_display, r_arm)

	# Display the result
	cv2.imshow("output", image_display)
	key = cv2.waitKey(1)
	if key == 27: # ESC
		break
	if key == ord("p"):
		while True:
			key = cv2.waitKey(1)
			if key == ord("p"): break

cap.release()
cv2.destroyAllWindows()

