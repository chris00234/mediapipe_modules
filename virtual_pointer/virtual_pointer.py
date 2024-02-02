import cv2
import numpy as np
import time
import os
import sys
sys.path.insert(1, '/Users/chris/Documents_local/CV/advanced_cv/Hand_tracking')
import hand_tracking_module as htm

####################################
#SELECTION MODE: TWO FINGERS UP#####
#DRAWING MODE: INDEX FINGER UP######

folderpath = '/Users/chris/Documents_local/CV/advanced_cv/virtual_pointer/Header'
mylist = os.listdir(folderpath)
overlay_list = []

for impath in mylist:
    img = cv2.imread(f'{folderpath}/{impath}')
    overlay_list.append(img)


header = overlay_list[3]
draw_color = (0,0,255)
brush_thickness = 10
eraser_thickness = 20

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon= 0.85)
xp,yp = 0,0
canvas = np.zeros((720,1280,3),np.uint8)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    lm_list = detector.findPosition(frame, draw = False)
    if len(lm_list) != 0:
        #tip of index finger
        x1, y1 = lm_list[8][1:]
        
        #tip of middle finger
        x2, y2 = lm_list[12][1:]
        fingers = detector.fingersUp()
        # print(fingers)
        
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            # print("SELECTION MODE")
            if y1 < 96:
                if x1 >= 250 and x1 <= 400:
                    header = overlay_list[3]
                    draw_color = (0, 0, 255)
                
                elif x1 >= 525 and x1 <= 660:
                    header = overlay_list[1]
                    draw_color = (0, 255, 0)
                
                elif x1 >= 780 and x1 <= 920:
                    header = overlay_list[2]
                    draw_color = (255, 0, 0)
                
                elif x1 >= 1050 and x1 <= 1100:
                    header = overlay_list[0]
                    draw_color = (0, 0, 0)
            cv2.rectangle(frame, (x1,y1-25), (x2,y2 + 25), draw_color, cv2.FILLED)
        
        if fingers[1] and fingers[2] == 0:
            cv2.circle(frame, (x1,y1), 10, draw_color, cv2.FILLED)
            # print("DRAWING MODE")
            if xp == 0 and yp == 0:
                xp,yp = x1, y1
            if draw_color == (0,0,0):
                cv2.line(frame, (xp,yp), (x1,y1), draw_color, eraser_thickness)
                cv2.line(canvas, (xp,yp), (x1,y1), draw_color, eraser_thickness)
            else:
                cv2.line(frame, (xp,yp), (x1,y1), draw_color, brush_thickness)
                cv2.line(canvas, (xp,yp), (x1,y1), draw_color, brush_thickness)
            xp,yp = x1,y1
            
    frame_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, frame_inv = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY_INV)
    frame_inv = cv2.cvtColor(frame_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, frame_inv)
    frame = cv2.bitwise_or(frame, canvas)
    #header img setup
    frame[:96, :] = header
    # frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow('img', frame)
    # cv2.imshow('canvas', canvas)
    cv2.waitKey(1)