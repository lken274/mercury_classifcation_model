import glob
import cv2
import numpy

video_dir = '/home/logan/Desktop/tf_models/blemish_detector/evaluation_videos/sept_2020_class3_dark_blemish/*.jpg'
window_name = 'image'
skip_frames = 10
low_Hue, low_Sat, low_Val = 0, 0, 54
high_Hue, high_Sat, high_Val = 41, 255, 255
minFruitSize = 10000

#load video from file
frames = [cv2.imread(file) for file in sorted(glob.glob(video_dir))]
frames = frames[skip_frames:]
for roiFrame in frames:
    #cv2.imshow(window_name, frame)
    #cv2.waitKey()
    frame_HSV = cv2.cvtColor(roiFrame, cv2.COLOR_BGR2HSV) #convert to HSV
    frame_HSV = cv2.medianBlur(frame_HSV, 5) #blur image
    frame_threshold = cv2.inRange(frame_HSV, (low_Hue, low_Sat, low_Val), (high_Hue, high_Sat, high_Val))
    retval, thr = cv2.threshold(frame_threshold, 25, 255, cv2.THRESH_BINARY); #threshold grey
    contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#find contours

    fruitHulls = []
    for contour in contours:
        area = cv2.contourArea(contour, False) #get contouring area to cull bad contours
        if (area < minFruitSize):
            continue
        x,y,w,h = cv2.boundingRect(contour)
        #check if fruit istouching boundary
        height, width, channels = roiFrame.shape
        if (x == 0 or (x+w) >= (width - 1)):
            continue

        #create a sparse hull from dense fruit contour
        numPts = len(contour)
        index, stride, offset = 0, 20, 0
        numSparsePtrs = numPts/stride
        sparseHull = []
        j = 0
        while (j < (numPts - stride)):
            if (index < numPts):
                sparseHull.append(contour[j])
            index = index + 1
            j = j + stride

        #find the largest convex hull that fits in our sparse hull
        sparseConvexHull = cv2.convexHull(sparseHull)
        #create a mask based on hull contours, (drawContours) [friutHullMask]
        #copy original frame to trimmed_BGR using friutHullMask as mask
        #Mat fruitCropBGR = trimmed_BGR(hullRects[i]);
        #Mat fruitCropHue = trimmed_Hue(hullRects[i]);
        #Mat localMask = friutHullMask(hullRects[i]);
        #Mat adp_threshold;
        #cv::GaussianBlur(fruitCropHue, fruitCropHue, Size(19, 19), 0, 0);
        #adaptiveThreshold(fruitCropHue, adp_threshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 73, 2);
        #minDefectSize = 200;
        #cv::bitwise_and(localMask, adp_threshold, adp_threshold);
        #cv::findContours(adp_threshold, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        #drawContours(fruitCropBGR, contours, i, color_black, 2, 8, hierarchy1);

#get list of every fruit and their original coordinates

#run classifier on each fruit in frame

#draw original frame
#place bounding boxes on original image