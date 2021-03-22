import glob
import cv2
import numpy as np
import time
import tensorflow as tf
from inferenceutils import *

video_dir = '/home/logan/Desktop/tf_models/blemish_detector/evaluation_videos/sept_2020_class3_dark_blemish/*.jpg'
model_dir = 'inference_graph/saved_model'
save_name = "demo_output/demo_single_1.avi"
labelmap_path = 'dataset/labelmap.pbtxt'
save_video = True

render_scale = 0.25 #resolution rescaling during edge detection to improve performance
skip_frames = 0 #start at a later point in the video so we aren't waiting for fruit
low_Hue, low_Sat, low_Val = 10, 80, 95
high_Hue, high_Sat, high_Val = 70, 255, 255
light_fruit_colour = (high_Hue,high_Sat,high_Val)
dark_fruit_colour = (low_Hue,low_Sat,low_Val)

minFruitSize = 10000 * render_scale
minAspectRatio = 0.5
detection_threshold = 0.5

def main():
    #load video from file
    frames = [cv2.imread(file) for file in sorted(glob.glob(video_dir))]
    frames = frames[skip_frames:]
    numFrames = len(frames) - skip_frames
    size = (frames[0].shape[1], frames[0].shape[0])
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
    model = tf.saved_model.load(model_dir)

    if (save_video == True):
        out_vid = cv2.VideoWriter(save_name,cv2.VideoWriter_fourcc(*'DIVX'), 10, (size))
   

    for idx,roiFrame in enumerate(frames):
        print("Frame " + str(idx) + " of " + str(numFrames))
        tf.keras.backend.clear_session()
        roi_resized = cv2.resize(roiFrame, (round(render_scale*roiFrame.shape[1]), round(render_scale*roiFrame.shape[0])))
        frame_HSV = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV) #convert to HSV
        frame_blur = cv2.bilateralFilter(frame_HSV, 5, 100, 100)
        frame_threshold = cv2.inRange(frame_blur, dark_fruit_colour, light_fruit_colour)
        edges = cv2.Canny(frame_threshold, 200, 250)
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]#find contours
        
        cnn_outputs = []
        for contour in contours:
            area = cv2.contourArea(contour, False) #get contouring area to cull bad contours
            if (area < minFruitSize):
                continue
            x,y,w,h = cv2.boundingRect(contour)
            if(w > (h / minAspectRatio) or h > (w / minAspectRatio)):
                continue
            #check if fruit istouching boundary
            height, width, channels = roi_resized.shape
            if (x == 0 or (x+w) >= (width - 1)):
                continue
            #create a sparse hull from dense fruit contour
            sparseHull = createSparseHull(contour, 1)
            #find the largest convex hull that fits in our sparse hull
            sparseConvexHull = cv2.convexHull(np.array(sparseHull, dtype=int))
            sparseConvexHull = np.array(sparseConvexHull).squeeze()
            x,y,w,h = cv2.boundingRect(sparseConvexHull)
            for point in sparseConvexHull:
                point[0] = round((point[0] - x) / render_scale)
                point[1] = round((point[1] - y) / render_scale)

            #create a mask based on hull contours, (drawContours) [fruitHullMask]
            origY, origH = round(y / render_scale), round(h / render_scale)
            origX, origW = round(x / render_scale), round(w / render_scale)
            fruitCropBGR = roiFrame[origY:origY+origH, origX:origX+origW]

            mask = np.zeros(fruitCropBGR.shape, dtype='uint8')
            cv2.drawContours(mask, [sparseConvexHull],-1,(255,255,255), cv2.FILLED)
            img2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            maskedImage = cv2.bitwise_and(fruitCropBGR, fruitCropBGR, mask=img2gray)

            output_dict = run_inference_for_single_image(model, maskedImage)

            boxes = output_dict['detection_boxes']
            #translate coordinate system from percentage of masked image to absolute full image
            for idx, pos in enumerate(boxes): 
                ymin = pos[0] * origH + origY
                ymax = pos[2] * origH + origY
                xmin = pos[1] * origW + origX
                xmax = pos[3] * origW + origX
                boxes[idx] = [ymin, xmin, ymax, xmax]

            cnn_outputs.append(output_dict)
            vis_util.visualize_boxes_and_labels_on_image_array(
            roiFrame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=False,
            line_thickness=3,
            min_score_thresh=detection_threshold)

        if (save_video == True):
            out_vid.write(roiFrame)
    if(save_video == True):
        out_vid.release()

def createSparseHull(contour, stride):
    if (stride == 1):
        return contour
    numPts = len(contour)
    numSparsePtrs = numPts/stride
    sparseHull = []
    index = 0
    j = 0
    while (j < (numPts - stride)):
        if (index < numPts):
            sparseHull.append(contour[j])
        index = index + 1
        j = j + stride
    return sparseHull

if __name__ == "__main__":
    main()
