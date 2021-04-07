import glob
import cv2
import numpy as np
import time
import tensorflow as tf
from natsort import natsorted
import json
import requests
import base64
import multiprocessing

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

video_dir = '/home/logan/Desktop/tf_models/blemish_detector/evaluation_videos/sequence_VIS_C3_blemish_03/*.jpg'
model_dir = 'inference_graph/saved_model/1'
save_name = "demo_output/sequence_VIS_C3_blemish_03_sensitive_serving.avi"
labelmap_path = 'dataset/labelmap.pbtxt'
grayscale = False
save_video = False
show_mask = False

render_scale = 0.33 #resolution rescaling during edge detection to improve performance
skip_frames = 0 #start at a later point in the video so we aren't waiting for fruit
low_Hue, low_Sat, low_Val = 0, 50, 65
high_Hue, high_Sat, high_Val = 80, 255, 255
light_fruit_colour = (high_Hue,high_Sat,high_Val)
dark_fruit_colour = (low_Hue,low_Sat,low_Val)

minFruitSize = 10000 * render_scale
minAspectRatio = 0.5
detection_threshold = 0.35

num_boxes_per_frame = 10

num_batch_frames = 1

def main():
    #load video from file
    tf.keras.backend.clear_session()
    frames = [cv2.imread(file) for file in natsorted(glob.glob(video_dir))]
    frames = frames[skip_frames:]
    numFrames = len(frames) - skip_frames
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
    query_thread = None
    manager = multiprocessing.Manager()
    tf_response_q = manager.Queue()

    first_cycle = True

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

        edges = cv2.Canny(frame_threshold, 250, 300)
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]#find contours

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
        

        if (query_thread != None):
            query_thread.join()

        if (tf_response_q.empty() == False):
            print("Fps: " + str(1.0 / (time.time() - start_time) * num_batch_frames))
            output_response = tf_response_q.get()
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
                bbox = previous_boundingBoxes[current_fruit_image]
                ymin = pos[0] * bbox[3] + bbox[1]
                ymax = pos[2] * bbox[3] + bbox[1]
                xmin = pos[1] * bbox[2] + bbox[0]
                xmax = pos[3] * bbox[2] + bbox[0]
                boxes[idx] = [ymin, xmin, ymax, xmax]
            boxes = np.array(boxes)
            vis_util.visualize_boxes_and_labels_on_image_array(
            prev_roiFrame,
            boxes,
            np.array(classes, dtype=np.uint8),
            scores,
            category_index,
            instance_masks=masked_reframed,
            use_normalized_coordinates=False,
            line_thickness=3,
            min_score_thresh=detection_threshold)

            print("Sending " + str(len(frame_images)))
            start_time = time.time()
            query_thread = multiprocessing.Process(name='tf_serving', target=query_serving, args=(frame_images, stub, tf_response_q))
            query_thread.start()
            previous_boundingBoxes = boundingBoxes[:]
            prev_roiFrame = np.copy(roiFrame)
            frame_images.clear()
            batch_count = 0

            if (save_video == True):
                out_vid.write(prev_roiFrame)
                
        if (first_cycle == True):
            first_cycle = False
            print("Sending " + str(len(frame_images)))
            start_time = time.time()
            query_thread = multiprocessing.Process(name='tf_serving', target=query_serving, args=(frame_images, stub, tf_response_q))
            query_thread.start()
            previous_boundingBoxes = boundingBoxes[:]
            prev_roiFrame = np.copy(roiFrame)
            frame_images.clear()
            batch_count = 0
   
    if(save_video == True):
        out_vid.release()

def get_float_vals(result):
    return list(result.float_val)

def query_serving(maskedImageList, stub, dataq):        
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
    dataq.put(result)
    return

if __name__ == "__main__":
    main()
