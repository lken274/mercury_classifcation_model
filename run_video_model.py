import glob
import cv2
import numpy as np
import time
import tensorflow as tf
from natsort import natsorted
import json
import requests
import base64

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

video_dir = '/home/logan/Desktop/tf_models/blemish_detector/evaluation_videos/VIS_fresh_cuts_Gold_05/*.jpg'
model_dir = 'inference_graph/saved_model/resnet101/1/'
save_name = "demo_output/VIS_fresh_cuts_Gold_05.avi"
labelmap_path = 'dataset/labelmap.pbtxt'
grayscale = False
save_video = True
show_mask = False

render_scale = 0.5 #resolution rescaling during edge detection to improve performance
skip_frames = 0 #start at a later point in the video so we aren't waiting for fruit
low_Hue, low_Sat, low_Val = 0, 48, 62
high_Hue, high_Sat, high_Val = 80, 255, 255
#low_Hue, low_Sat, low_Val = 0, 62, 82
#high_Hue, high_Sat, high_Val = 80, 255, 255
light_fruit_colour = (high_Hue,high_Sat,high_Val)
dark_fruit_colour = (low_Hue,low_Sat,low_Val)

minFruitSize = 10000 * render_scale
minAspectRatio = 0.5
detection_threshold = 0.3

num_boxes_per_frame = 100

num_batch_frames = 1

def main():
    #load video from file
    tf.keras.backend.clear_session()
    frames = [cv2.imread(file) for file in natsorted(glob.glob(video_dir))]
    frames = frames[skip_frames:]
    numFrames = len(frames)
    size = (frames[0].shape[1], frames[0].shape[0])
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))

    label = ['blemish']
    GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3 * 2
    GRPC_MAX_SEND_MESSAGE_LENGTH = 4096 * 4096 * 3 * 2
    channel = grpc.insecure_channel('0.0.0.0:8500', options=[('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH), ('grpc.max_send_message_length', GRPC_MAX_SEND_MESSAGE_LENGTH)])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    if (save_video == True):
        out_vid = cv2.VideoWriter(save_name,cv2.VideoWriter_fourcc(*'DIVX'), 10, (size))

    frame_images = []
    batch_count = 0
    for idx,roiFrame in enumerate(frames):
        print("Frame " + str(idx) + " of " + str(numFrames))
        roi_resized = cv2.resize(roiFrame, (round(render_scale*roiFrame.shape[1]), round(render_scale*roiFrame.shape[0])))
        frame_blur = cv2.medianBlur(roi_resized, 5)
        frame_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(frame_lab)
        cla = clahe.apply(l)
        frame_blur_lab = cv2.merge((cla,a,b))
        frame_blur = cv2.cvtColor(frame_blur_lab, cv2.COLOR_LAB2BGR)
        
        if (grayscale == False):
            frame_HSV = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV) #convert to HSV
            frame_threshold = cv2.inRange(frame_HSV, dark_fruit_colour, light_fruit_colour)
            height, width, channels = frame_blur.shape
        else:
            grayFrame = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
            frame_threshold = cv2.adaptiveThreshold(grayFrame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)
            frame_threshold = cv2.bitwise_not(frame_threshold)
            height, width = grayFrame.shape

        contours = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]#find contours

        boundingBoxes = []

        for contour in contours:
            contour = cv2.approxPolyDP(contour,0.004*cv2.arcLength(contour,True),True)
            area = cv2.contourArea(contour, False) #get contouring area to cull bad contours

            if (area < minFruitSize):
                continue

            x,y,w,h = cv2.boundingRect(contour)
            if(w > (h / minAspectRatio) or h > (w / minAspectRatio)):
                continue
            #check if fruit istouching boundary
            if (x == 0 or (x+w) >= (width - 1)):
                continue

            contour = contour.squeeze()
            x,y,w,h = cv2.boundingRect(contour)

            for point in contour:
                point[0] = round((point[0] - x) / render_scale)
                point[1] = round((point[1] - y) / render_scale)

            #create a mask based on hull contours, (drawContours) [fruitHullMask]
            origY, origH = round(y / render_scale), round(h / render_scale)
            origX, origW = round(x / render_scale), round(w / render_scale)
            boundingBoxes.append((origX, origY, origW, origH))
            fruitCropBGR = roiFrame[origY:origY+origH, origX:origX+origW]

            mask = np.zeros(fruitCropBGR.shape, dtype='uint8')
            cv2.drawContours(mask, [contour],-1,(255,255,255), cv2.FILLED)
            img2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            maskedImage = cv2.bitwise_and(fruitCropBGR, fruitCropBGR, mask=img2gray)
            frame_images.append(maskedImage)
            if (show_mask == True):
                cv2.imshow('masked', maskedImage)
                cv2.waitKey(10)
        #accumulate all images from this frame to send
        if (len(frame_images) == 0):
            continue

        batch_count = batch_count + 1
        if (batch_count < num_batch_frames):
            continue
        
        print("Sending " + str(len(frame_images)))
        start_time = time.time()
        output_response = query_serving(frame_images, stub)
        print("Fps: " + str(1.0 / (time.time() - start_time) * num_batch_frames))
        frame_images.clear()
        batch_count = 0

        boxes_raw = get_float_vals(output_response.outputs['detection_boxes'])
        boxes = []
        counter = 0
        one_box = []
        for coord in boxes_raw:
            counter = counter + 1
            one_box.append(coord)
            if (counter == 4):
                boxes.append(tuple(one_box))
                one_box.clear()
                counter = 0
        classes = get_float_vals(output_response.outputs['detection_classes'])
        scores = get_float_vals(output_response.outputs['detection_scores'])
        #masked_reframed = get_float_vals(output_response.outputs['detection_masks_reframed'])
        masked_reframed = None
        #translate coordinate system from percentage of masked image to absolute full image
        for idx, pos in enumerate(boxes): 
            current_fruit_image = int(idx / num_boxes_per_frame) 
            bbox = boundingBoxes[current_fruit_image]
            ymin = pos[0] * bbox[3] + bbox[1]
            ymax = pos[2] * bbox[3] + bbox[1]
            xmin = pos[1] * bbox[2] + bbox[0]
            xmax = pos[3] * bbox[2] + bbox[0]
            boxes[idx] = [ymin, xmin, ymax, xmax]

        boxes,scores = cull_below_threshold(boxes, scores, detection_threshold)
        boxes = np.array(boxes)
        scores = np.array(scores)
        #boxes, scores = non_max_suppression_fast(boxes, scores, 0.2)
        boxes = np.array(boxes)
        scores = np.array(scores)

        vis_util.visualize_boxes_and_labels_on_image_array(
        roiFrame,
        boxes,
        np.array(classes, dtype=np.uint8),
        scores,
        category_index,
        instance_masks=masked_reframed,
        use_normalized_coordinates=False,
        line_thickness=3,
        min_score_thresh=detection_threshold)

        if (save_video == True):
            out_vid.write(roiFrame)
    if(save_video == True):
        out_vid.release()

def get_float_vals(result):
    return list(result.float_val)

def query_serving(maskedImageList, stub):        
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = 'blemish_detector'
    grpc_request.model_spec.signature_name = 'serving_default'
    image_data = []
    
    for image in maskedImageList:
        buffer = cv2.imencode('.png', image)[1].tostring()
        image_data.append(buffer)
    grpc_request.inputs["input_tensor"].CopyFrom(
    tf.make_tensor_proto(image_data, dtype=tf.dtypes.string, shape=[len(image_data)]))
    result = stub.Predict(grpc_request, 5)  # 10 secs timeout
    return result

def cull_below_threshold(boxes, scores, thresh):
    new_boxes = []
    new_scores = []
    for idx,score in enumerate(scores):
        if (score >= thresh):
            new_boxes.append(boxes[idx])
            new_scores.append(score)
    return (new_boxes, new_scores)

def non_max_suppression_fast(boxes, scores, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return ([],[])
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    y1 = boxes[:,0]
    x1 = boxes[:,1]
    y2 = boxes[:,2]
    x2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return (boxes[pick].astype("int"), scores[pick].astype("float"))

if __name__ == "__main__":
    main()