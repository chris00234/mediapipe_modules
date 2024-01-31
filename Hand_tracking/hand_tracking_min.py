import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prev_time = 0
current_time = 0

while True:
    ret, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #get hand finger with landmark
            for id, lm in enumerate(handLms.landmark):
                h,w,ch = img.shape
                #get center x and center y
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                #wrist
                if id == 0:
                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)
                #tip of Thumb
                if id == 4:
                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)
                    
            #draw hand landmark connections 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    #showing fps
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time
    cv2.putText(img, 'fps = ' + str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)