import cv2
import mediapipe as mp
import time


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius= 1)

cap = cv2.VideoCapture(0)
current_time = 0
prev_time = 0
while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, landmark_drawing_spec = drawSpec)

            for id,lm in enumerate(faceLms.landmark):
                h, w, c = img.shape
                x,y = int(lm.x * w), int(lm.y * h)
                    
    
    #Showing frame
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time
    cv2.putText(img, 'fps = ' + str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('image', img)
    cv2.waitKey(1)