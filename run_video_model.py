import glob
import cv2
import numpy as np

video_dir = '/home/logan/Desktop/tf_models/blemish_detector/evaluation_videos/sept_2020_class3_dark_blemish/*.jpg'
window_name = 'image'
skip_frames = 100
low_Hue, low_Sat, low_Val = 10, 75, 90
high_Hue, high_Sat, high_Val = 70, 255, 255
light_fruit_colour = (high_Hue,high_Sat,high_Val)
dark_fruit_colour = (low_Hue,low_Sat,low_Val)

minFruitSize = 10000
minAspectRatio = 0.5

#load video from file
frames = [cv2.imread(file) for file in sorted(glob.glob(video_dir))]
frames = frames[skip_frames:]
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
for roiFrame in frames:
    frame_HSV = cv2.cvtColor(roiFrame, cv2.COLOR_BGR2HSV) #convert to HSV

    h,s,v = cv2.split(frame_HSV)
    cl1 = clahe.apply(v)
    cl2 = clahe.apply(s)
    normalised = cv2.merge((h,cl2,cl1))
    normalised_bgr = cv2.cvtColor(normalised, cv2.COLOR_HSV2BGR)
    frame_blur = cv2.bilateralFilter(normalised, 50, 50, 50)
    normalised_blur = cv2.cvtColor(frame_blur, cv2.COLOR_HSV2BGR)
    frame_threshold = cv2.inRange(frame_blur, dark_fruit_colour, light_fruit_colour)
    edges = cv2.Canny(frame_threshold, 200, 250)
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]#find contours

    fruitHulls = []
    for contour in contours:
        area = cv2.contourArea(contour, False) #get contouring area to cull bad contours
        if (area < minFruitSize):
            continue
        x,y,w,h = cv2.boundingRect(contour)
        if(w > (h / minAspectRatio) or h > (w / minAspectRatio)):
            continue
        #check if fruit istouching boundary
        height, width, channels = roiFrame.shape
        if (x == 0 or (x+w) >= (width - 1)):
            continue
        #create a sparse hull from dense fruit contour
        numPts = len(contour)
        index, stride, offset = 0, 10, 0
        numSparsePtrs = numPts/stride
        sparseHull = []
        j = 0
        while (j < (numPts - stride)):
            if (index < numPts):
                sparseHull.append(contour[j])
            index = index + 1
            j = j + stride

        #find the largest convex hull that fits in our sparse hull
        sparseConvexHull = cv2.convexHull(np.array(sparseHull, dtype=int))
        sparseConvexHull = np.array(sparseConvexHull).squeeze()
        x,y,w,h = cv2.boundingRect(sparseConvexHull)
        for point in sparseConvexHull:
            point[0] = point[0] - x
            point[1] = point[1] - y

        fruitHulls.append((sparseConvexHull, (x,y,w,h)))
        #create a mask based on hull contours, (drawContours) [fruitHullMask]
        fruitCropBGR = roiFrame[y:y+h, x:x+w]
        fruitCropHue = cv2.cvtColor(fruitCropBGR, cv2.COLOR_BGR2HSV)
        #cv2.imshow('cropped', fruitCropBGR)

        mask = np.zeros(fruitCropBGR.shape, dtype='uint8')
        cv2.drawContours(mask, [sparseConvexHull],-1,(255,255,255), cv2.FILLED)
        img2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        cv2.imshow('outlined', img2gray)

        maskedImage = cv2.bitwise_and(fruitCropBGR, fruitCropBGR, mask=img2gray)
 
        cv2.imshow(window_name, maskedImage)
        cv2.waitKey(10)

#get list of every fruit and their original coordinates

#run classifier on each fruit in frame

#draw original frame
#place bounding boxes on original image