import glob
import cv2
import numpy as np
import time
import tensorflow as tf
from inferenceutils import *
from natsort import natsorted

video_dir = '/home/logan/Desktop/tf_models/blemish_detector/evaluation_videos/sequence_VIS_C3_blemish_03/*.jpg'
model_dir = 'inference_graph/saved_model'
save_name = "demo_output/sequence_VIS_C3_blemish_03_sensitive.avi"
labelmap_path = 'dataset/labelmap.pbtxt'
grayscale = False
save_video = True
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

def main():
    #load video from file
    frames = [cv2.imread(file) for file in natsorted(glob.glob(video_dir))]
    frames = frames[skip_frames:]
    numFrames = len(frames) - skip_frames
    size = (frames[0].shape[1], frames[0].shape[0])
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
    model = tf.saved_model.load(model_dir)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))

    if (save_video == True):
        out_vid = cv2.VideoWriter(save_name,cv2.VideoWriter_fourcc(*'DIVX'), 10, (size))
   

    for idx,roiFrame in enumerate(frames):
        print("Frame " + str(idx) + " of " + str(numFrames))
        start_time = time.time()
        tf.keras.backend.clear_session()
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

        cnn_outputs = []
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
            fruitCropBGR = roiFrame[origY:origY+origH, origX:origX+origW]

            mask = np.zeros(fruitCropBGR.shape, dtype='uint8')
            cv2.drawContours(mask, [contour],-1,(255,255,255), cv2.FILLED)
            img2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            maskedImage = cv2.bitwise_and(fruitCropBGR, fruitCropBGR, mask=img2gray)
            if (show_mask == True):
                cv2.imshow('masked', maskedImage)
                cv2.waitKey(10)
            start_time = time.time()
            output_dict = run_inference_for_single_image(model, maskedImage)
            print("FPS: " + str((1 / (time.time() - start_time))))
        
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

if __name__ == "__main__":
    main()
