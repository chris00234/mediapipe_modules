import cv2
import mediapipe as mp
import time
import hand_tracking_module as htm


cap = cv2.VideoCapture(0)
prev_time = 0
current_time = 0
detector = htm.handDetector()
while True:
    ret, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])
    #showing fps
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time
    cv2.putText(img, 'fps = ' + str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)