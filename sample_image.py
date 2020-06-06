import numpy as np
import scipy.ndimage
import scipy.misc
import cv2
from bodypix.pose_engine import PoseEngine, EDGES, BODYPIX_PARTS


def clip_heatmap(heatmap, v0, v1):
	a = v0 / (v0 - v1);
	b = 1.0 / (v1 - v0);
	return np.clip(a + b * heatmap, 0.0, 1.0);

RED_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "right" in v]
GREEN_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "hand" in v or "torso" in v]
BLUE_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "leg" in v or "arm" in v or "face" in v or "hand" in v]

# Initialize
model = 'bodypix/models/bodypix_mobilenet_v1_075_640_480_16_quant_edgetpu_decoder.tflite'
engine = PoseEngine(model)
inference_size = (engine.image_width, engine.image_height)

# Read image and pre-process
image_org = cv2.imread("hayato4M3A0906_TP_V.jpg")
image = cv2.resize(image_org, inference_size)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.uint8)

# Run inference
inference_time, poses, heatmap, bodyparts = engine.DetectPosesInImage(image)

# Post-process
heatmap = clip_heatmap(heatmap,  -1.0,  1.0)
rgb_heatmap = np.dstack([
	heatmap*(np.sum(bodyparts[:,:,RED_BODYPARTS], axis=2)-0.5)*100,
	heatmap*(np.sum(bodyparts[:,:,GREEN_BODYPARTS], axis=2)-0.5)*100,
	heatmap*(np.sum(bodyparts[:,:,BLUE_BODYPARTS], axis=2)-0.5)*100,
])

rgb_heatmap = 155 * np.clip(rgb_heatmap, 0, 1)
rescale_factor = [
	image_org.shape[0] / heatmap.shape[0],
	image_org.shape[1] / heatmap.shape[1],
	1
]

rgb_heatmap = scipy.ndimage.zoom(rgb_heatmap, rescale_factor, order=0)
output_image = image_org + rgb_heatmap
int_img = np.uint8(np.clip(output_image, 0, 255))

# Display the result
cv2.imshow("output", int_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
