import cv2
import imageio
import glob
from os.path import join
import pdb
import numpy as np
from pathlib import Path
# image_list = glob.glob(join('/data/ischakra/synthetic/gundam-gray/images/', '*.jpg'))
# output_path = '/data/ischakra/synthetic/gundam-gray/video.mp4'
# image_list.sort()

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# writer = None
# (h, w) = (None, None)
# zeros = None

# for image_file in image_list:
# 	img = cv2.imread(image_file)
# 	img = cv2.resize(img, (504, 378))
# 	if writer is None:
# 		# store the image dimensions, initialize the video writer,
# 		# and construct the zeros array
# 		(h, w) = img.shape[:2]
# 		writer = cv2.VideoWriter(output_path, fourcc, 5,(w, h), True)

# 	# write the output frame to file
# 	writer.write(img)
# writer.release()

image_list = glob.glob(join('/home/ischakra/src/code/nerf_pl/results/llff/gundam-gray/', '*.png'))
output_path = '/home/ischakra/src/code/nerf_pl/results/llff/gundam-gray/video.mp4'
image_list.sort()

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = None
(h, w) = (None, None)
zeros = None
image_list = image_list[15:90]

for image_file in image_list:
	img = cv2.imread(image_file)
	img = cv2.resize(img, (504, 378))
	depth_name = 'depth_' + Path(image_file).stem + '.pfm'
	depth_path = str(Path(image_file).parents[0])
	depth = cv2.imread(join(depth_path, depth_name),-1)
	depth = cv2.resize(depth, (504, 378))

	minimum_brightness = 0.4
	brightness = np.sum(depth) / (255 * 504 * 378)
	ratio = brightness / minimum_brightness
	if ratio < 1:
		depth = cv2.convertScaleAbs(depth, alpha = 1 / ratio, beta = 0)
	# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
	depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
	out = cv2.hconcat([img, depth_colormap])
	# cv2.imwrite('out.png', out)
	# pdb.set_trace()
	if writer is None:
		# store the image dimensions, initialize the video writer,
		# and construct the zeros array
		(h, w) = img.shape[:2]
		writer = cv2.VideoWriter(output_path, fourcc, 15,(w*2, h), True)

	# write the output frame to file
	writer.write(out)
writer.release()



