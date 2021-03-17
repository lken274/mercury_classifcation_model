import glob
import cv2
import numpy as np
import time

video_dir = '/home/logan/Desktop/tf_models/blemish_detector/evaluation_videos/sept_2020_class3_dark_blemish/*.jpg'
window_name = 'image'
render_scale = 0.25
skip_frames = 100
low_Hue, low_Sat, low_Val = 10, 80, 95
high_Hue, high_Sat, high_Val = 70, 255, 255
light_fruit_colour = (high_Hue,high_Sat,high_Val)
dark_fruit_colour = (low_Hue,low_Sat,low_Val)

minFruitSize = 10000 * render_scale
minAspectRatio = 0.5

def main():
    #load video from file
    frames = [cv2.imread(file) for file in sorted(glob.glob(video_dir))]
    frames = frames[skip_frames:]
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
    for roiFrame in frames:
        roi_resized = cv2.resize(roiFrame, (round(render_scale*roiFrame.shape[1]), round(render_scale*roiFrame.shape[0])))
        frame_HSV = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV) #convert to HSV
        frame_blur = cv2.bilateralFilter(frame_HSV, 5, 100, 100)
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

            fruitHulls.append((sparseConvexHull, (x,y,w,h)))
            #create a mask based on hull contours, (drawContours) [fruitHullMask]
            origY, origH = round(y / render_scale), round(h / render_scale)
            origX, origW = round(x / render_scale), round(w / render_scale)
            fruitCropBGR = roiFrame[origY:origY+origH, origX:origX+origW]

            mask = np.zeros(fruitCropBGR.shape, dtype='uint8')
            cv2.drawContours(mask, [sparseConvexHull],-1,(255,255,255), cv2.FILLED)
            img2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            maskedImage = cv2.bitwise_and(fruitCropBGR, fruitCropBGR, mask=img2gray)
            cv2.imshow('cropped', fruitCropBGR)
            cv2.imshow('outlined', img2gray)
            cv2.imshow('masked', maskedImage)
            cv2.waitKey(100)
    #get list of every fruit and their original coordinates

    #run classifier on each fruit in frame

    #draw original frame
    #place bounding boxes on original image

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
