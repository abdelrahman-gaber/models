import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import cv2 
import math
import argparse

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the code is stored in the object_detection folder.
sys.path.append("..")
sys.path.append("../..")

from utils import label_map_util
from utils import visualization_utils as vis_util


#MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('../data', 'mscoco_label_map.pbtxt')
#NUM_CLASSES = 90

# helper function
def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Load a (frozen) Tensorflow model into memory.
#detection_graph = tf.Graph()
#with detection_graph.as_default():
	#od_graph_def = tf.GraphDef()
	#with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		#serialized_graph = fid.read()
		#od_graph_def.ParseFromString(serialized_graph)
		#tf.import_graph_def(od_graph_def, name='')

# Loading label map
#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(2, 3) ]



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-images', help = 'Full path to original images', required = True)
	parser.add_argument('-output', help = 'Full path to output results images', required = True )
	parser.add_argument('-thresh', help = 'Threshold to visualize detections', required = True )
	parser.add_argument('-model', help = 'full name of the model folder', required = True )

	args = vars(parser.parse_args())

	if args['images'] is not None:
		images_path = args['images']
	if args['output'] is not None:
		output_path = args['output']
	if args['thresh'] is not None:
                Threshold = float(args['thresh'])
	if args['model'] is not None:
                MODEL_NAME = args['model']

	#MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
	#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'

	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
	# List of the strings that is used to add correct label for each box.
	PATH_TO_LABELS = os.path.join('../data', 'mscoco_label_map.pbtxt')
	NUM_CLASSES = 90


	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	# Loading label map
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			# Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			#for image_path in TEST_IMAGE_PATHS:
			for files in os.scandir(images_path):
				if files.is_file() and (files.name.endswith('.jpg') or files.name.endswith('.png')) :
					image_path = os.path.join(images_path, files.name )
					print(image_path)
					image = Image.open(image_path)
					#img = cv2.imread(image_path)
					save_txt_path = os.path.join(output_path, os.path.splitext(files.name)[0] + ".txt" )
					save_img_path = os.path.join(output_path, os.path.splitext(files.name)[0] + ".jpg" )
					# the array based representation of the image will be used later in order to prepare the
					# result image with boxes and labels on it.
					image_np = load_image_into_numpy_array(image)
					# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
					image_np_expanded = np.expand_dims(image_np, axis=0)
					# Actual detection.
					(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
					bboxes = np.squeeze(boxes)
					bscores = np.atleast_2d(np.squeeze(scores)).T
					bclasses = np.atleast_2d(np.squeeze(classes)).T
					#img = np.array(image)
					#img = img[:, :, ::-1].copy()
					f = open(save_txt_path, 'ab')
					im_width, im_height = image.size 
					for idx, row in enumerate(bboxes):
						if bclasses[idx] == 1: # people only 
							if bscores[idx] >= Threshold:
								#print("Person detected")
								#print(bboxes[idx])
								#print(bscores[idx])
								#print(bclasses[idx])
								y_min = int(bboxes[idx][0]*im_height)
								x_min = int(bboxes[idx][1]*im_width)
								y_max = int(bboxes[idx][2]*im_height)
								x_max = int(bboxes[idx][3]*im_width)
								#cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
								det_person =np.atleast_2d( np.asarray([x_min, y_min, x_max-x_min, y_max-y_min, bscores[idx] ]) ) # [detection prob]
								np.savetxt(f, det_person, fmt=["%d",]*4 + ["%1.3f"], delimiter=",")
					f.close
					#cv2.imwrite(save_img_path, img)
					
