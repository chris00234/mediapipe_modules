import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)

current_time = 0
prev_time = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)


while True:
    ret, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    
    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            #print(id, detection)
            #print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            h, w, c = img.shape
            bbox_c = detection.location_data.relative_bounding_box
            bbox = int(bbox_c.xmin * w), int(bbox_c.ymin * h), int(bbox_c.width * w), int(bbox_c.height * h)
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, 'score = ' + str(int(detection.score[0] * 100)) + '%', (bbox[0],bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            
    current_time = time.time()
    fps = 1/ (current_time - prev_time)
    prev_time = current_time
    cv2.putText(img, 'fps = ' + str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)