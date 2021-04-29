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
detection_threshold = 0.75
nonmax_sigma = 0.1

num_boxes_per_frame = 40

num_batch_frames = 1

def main():
    #load video from file
    tf.keras.backend.clear_session()
    frames = [cv2.imread(file) for file in natsorted(glob.glob(video_dir))]
    frames = frames[skip_frames:]
    numFrames = len(frames)
    size = (frames[0].shape[1], frames[0].shape[0])
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
    print(category_index)
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
            #contour = cv2.approxPolyDP(contour,0.004*cv2.arcLength(contour,True),True)
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
        

        boxes_raw = get_float_vals(output_response.outputs['detection_boxes'])
        #print("Box length: " + str(len(boxes_raw)))
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

        #separate receives values into different fruits
        all_fruit_data = (list(chunks(boxes, num_boxes_per_frame)),
                        list(chunks(classes, num_boxes_per_frame)), list(chunks(scores, num_boxes_per_frame)))

        for idx, pos in enumerate(all_fruit_data[0]): #iterate through each fruit on screen
            #b,s,c = cull_below_threshold(all_fruit_data[0][idx], all_fruit_data[2][idx], all_fruit_data[1][idx], detection_threshold)
            b,s,c = all_fruit_data[0][idx], all_fruit_data[2][idx], all_fruit_data[1][idx]
            for defectidx,defectpos in enumerate(b):
                bbox = boundingBoxes[idx]
                ymin = defectpos[0] * bbox[3] + bbox[1]
                ymax = defectpos[2] * bbox[3] + bbox[1]
                xmin = defectpos[1] * bbox[2] + bbox[0]
                xmax = defectpos[3] * bbox[2] + bbox[0]

                b[defectidx] = [ymin, xmin, ymax, xmax]

            b,s,c = np.array(b), np.array(s), np.array(c, dtype=np.uint8)
            keepIndexes = py_cpu_softnms(b,s,sigma=nonmax_sigma, thresh=detection_threshold)
            b,s,c = b[keepIndexes],s[keepIndexes],c[keepIndexes]

            vis_util.visualize_boxes_and_labels_on_image_array(
            roiFrame,
            b,
            c,
            s,
            category_index,
            instance_masks=masked_reframed,
            use_normalized_coordinates=False,
            line_thickness=3,
            min_score_thresh=detection_threshold)

        frame_images.clear()
        batch_count = 0

        if (save_video == True):
            out_vid.write(roiFrame)
    if(save_video == True):
        out_vid.release()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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

def cull_below_threshold(boxes, scores, classes, thresh):
    new_boxes = []
    new_scores = []
    new_classes = []
    for idx,score in enumerate(scores):
        if (score >= thresh):
            new_boxes.append(boxes[idx])
            new_scores.append(score)
            new_classes.append(classes[idx])
    return (new_boxes, new_scores, new_classes)

def py_cpu_softnms(dets, sc, Nt=0.5, sigma=1, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    if len(dets) == 0:
        return []
    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)

    return keep

if __name__ == "__main__":
    main()
