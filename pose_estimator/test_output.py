import cv2
import mediapipe as mp
import time
import pose_estimator_module as pem


cap = cv2.VideoCapture(0)
current_time = 0
prev_time = 0
detector = pem.poseDetector()

while True:
    ret, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    #highlight elbow
    cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), -1)
    #Showing frame
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time
    cv2.putText(img, 'fps = ' + str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("image", img)
    # if put 10, frame drops
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break


    cv2.imshow("image", img)
    # if put 10, frame drops
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break