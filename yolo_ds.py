# import the necessary packages
import os
import numpy as np
import datetime
from scipy import spatial
import cv2
from input_retrieval import *
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from tools import generate_detections as gdet

#All these classes will be counted as 'vehicles'
list_of_vehicles = ["car","motorbike","bus","truck"]
# Setting the threshold for the number of frames to search a vehicle for
inputWidth, inputHeight = 416, 416

#Parse command line arguments and extract the values required
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, logOutputPath, \
	preDefinedConfidence, preDefinedThreshold, USE_GPU= parseCommandLineArguments()

# Initialize a list of colors to represent each possible class label
np.random.seed(49)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

def initializeVideoWriter(video_width, video_height, videoStream):
	# Getting the fps of the source video
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	# initialize our video writer
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
		(video_width, video_height), True)

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Using GPU if flag is passed
if USE_GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = initializeVideoWriter(video_width, video_height, videoStream)
my_frame = 0

max_cosine_distance = 0.4
nn_budget = None

## Check for y to see if the vehicle has crossed that to count
y_check = int(video_height/2)

# initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric, n_init=3)

vehicleCount = 0
info_mp = {}
info_counted= []
detected = False
logfile = open(logOutputPath,"w")
print("Logfile:", logfile)
# loop over frames from the video file stream
while True:
	print("================NEW FRAME================")
	my_frame+= 1
	print("FRAME:\t", my_frame)

	# Initialization for each iteration
	boxes, confidences, classIDs = [], [], []

	# read the next frame from the file
	(grabbed, frame) = videoStream.read()

  # Draw line for which vehicle count is increments when crossed
	cv2.line(frame, (0, y_check), (video_width, y_check), (0, 0xFF, 0), 2)

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	for output in layerOutputs:
		for i, detection in enumerate(output):
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > preDefinedConfidence:

				box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))                          

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
		preDefinedThreshold)

	ids = []
	for i in range(len(boxes)): ids.append(classIDs[i])

	boxes_f = []
	scores_f = []
	class_f = []

	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
		  boxes_f.append(boxes[i])
		  scores_f.append(confidences[i])
		  class_f.append(ids[i])

		boxes_np = np.array(boxes_f)
		scores_np = np.array(scores_f)
		class_np = np.array(class_f)
		features = encoder(frame, boxes_np)  

	detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes_np, scores_np, class_np, features)]

	tracker.predict()
	tracker.update(detections)
	frame_time = datetime.datetime.now()
  # Draw line for which vehicle count is increments when crossed
	cv2.putText(frame, str(frame_time), (video_width-500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0xFF, 0), 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)
		
	## Vehicle counting code
	for t in tracker.tracks:
		crossed_hor_line = 0
    
		if not t.is_confirmed() or t.time_since_update > 1:
			continue
  
		if((t.track_id not in info_mp) and (LABELS[t.class_name] in list_of_vehicles)): 
			info_mp[t.track_id] = t.tlbr_midpoint(t.to_tlbr())
			continue

		if((t.track_id not in info_counted) and (t.track_id in info_mp)):
			mp_p = info_mp[t.track_id]
			mp = t.tlbr_midpoint(t.to_tlbr())
			crossed_hor_line = (mp_p[1] <= y_check < mp[1]) or (mp[1] <= y_check < mp_p[1])
			##print("Track", t.track_id, mp_p, mp, crossed_hor_line)

			if(crossed_hor_line): 
				info_counted.append(t.track_id)
				vehicleCount = vehicleCount + 1
				print("Count:", vehicleCount, "\ttime:", frame_time, "\tframe_id:", my_frame, "\ttrack_id:", t.track_id) 	
				print("Count:", vehicleCount, "\ttime:", frame_time, "\tframe_id:", my_frame, "\ttrack_id:", t.track_id, file=logfile) 	

		# Draw detection box
    # extract the bounding box coordinates
		(x, y) = (int(t.to_tlwh()[0]), int(t.to_tlwh()[1]))
		(w, h) = (int(t.to_tlwh()[2]), int(t.to_tlwh()[3]))
		color = [int(c) for c in COLORS[t.class_name]]
    ## add the bounding box
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
		# text of label and score on top of the bounding box
		text = "{}: {:.4f}".format(LABELS[t.class_name], t.detection_confidence)
		cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		## Track id and center dot in the middle of the bounding box
		cv2.putText(frame, str(t.track_id), (x + (w//2), y+ (h//2)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)
		
	# Display Vehicle Count if a vehicle has passed the line 
	cv2.putText(frame, 'Detected Vehicles: ' + str(vehicleCount), (20, 20), \
  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0xFF, 0), 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)

	writer.write(frame)
	print("Total Vehicle Count:", vehicleCount, "Tracker Length:", len(tracker.tracks))
	print("Info Counted", info_counted)
	#if my_frame == 30:
		#break	

# release the file pointers
print("[INFO] cleaning up...")
logfile.close()
writer.release()
videoStream.release()
