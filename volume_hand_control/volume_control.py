import cv2
import time
import numpy as np
import sys
import math
import osascript
import os
sys.path.insert(1, '/Users/chris/Documents_local/CV/advanced_cv/Hand_tracking')
import hand_tracking_module as htm
##################################
wcam, hcam = 640, 480
##################################



cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
prev_time = 0

detector = htm.handDetector(detectionCon= 0.7)

#for windows
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(
#     IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = interface.QueryInterface(IAudioEndpointVolume)
# # volume.GetMute()
# # volume.GetMasterVolumeLevel()
# vol_range = volume.GetVolumeRange()
# min_vol = vol_range[0]
# max_vol = vol_range[1]
# # volume.SetMasterVolumeLevel(-20.0, None)
def set_volume(volume_level):
    os.system(f"osascript -e 'set volume output volume {volume_level}'")

vol_bar = 400
vol_percentage = 0
while True:
    ret, frame = cap.read()
    
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    
    if len(lmList) != 0:
        #print(lmList[4], lmList[8])
        x1,y1 = lmList[4][1], lmList[4][2]
        x2,y2 = lmList[8][1], lmList[8][2]
        c1,c2 = (x1+x2)//2 , (y1+y2)//2
        
        cv2.circle(frame, (x1,y1), 10, (255, 0, 255), -1)
        cv2.circle(frame, (x2,y2), 10, (255, 0, 255), -1)    
        cv2.line(frame, (x1,y1), (x2,y2), (255, 0, 255), 3)
        cv2.circle(frame, (c1,c2), 7, (255, 0, 255), -1) 
        
        #length range => 50 to 300
        #volume range => -65 to 0
        #for windows
        length = math.hypot(x2-x1, y2-y1)
        # vol = np.interp(length, [50,300], [min_vol, max_vol])
        # volume.SetMasterVolumeLevel(vol, None)
        # print(length)
        
        #mac version:
        vol = int(np.interp(length, [50,300], [0, 100]))
        vol_bar = int(np.interp(length, [50,300], [400, 150]))
        vol_percentage = int(np.interp(length, [50,300], [0, 100]))
        # os.system('osascript -e "set Volume %d"' % vol)
        print(vol)
        set_volume(vol)
        if length < 50:
            cv2.circle(frame, (c1,c2), 7, (152, 153, 214), -1) 
        
    cv2.rectangle(frame, (50, 150), (85, 400), (215, 255, 91), 3)
    cv2.rectangle(frame, (50, vol_bar), (85, 400), (25, 255, 51), -1)
    cv2.putText(frame, f'Volume level: {int(vol_percentage)}%', (40,450), cv2.FONT_HERSHEY_COMPLEX, 1, (25, 255, 51), 3)
    
    ###FPS###
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f'FPS: {int(fps)}', (0,25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    #########
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)