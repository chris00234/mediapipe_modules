import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDectectionCon = 0.5):
        self.minDetectionCon = minDectectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                h, w, c = img.shape
                bbox_c = detection.location_data.relative_bounding_box
                bbox = int(bbox_c.xmin * w), int(bbox_c.ymin * h), int(bbox_c.width * w), int(bbox_c.height * h)
                bboxs.append([bbox, detection.score])
                # cv2.rectangle(img, bbox, (255,0,255), 2)
                if draw:
                    self.fancyDraw(img, bbox)
                    cv2.putText(img, 'score = ' + str(int(detection.score[0] * 100)) + '%', (bbox[0],bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        return img, bboxs
    
    def fancyDraw(self, img, bbox, l = 30, t = 7, rt = 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (255,0,255), thickness = rt)
        # Top left x,y
        cv2.line(img, (x,y), (x+l, y), (255, 0, 255), thickness= t)
        cv2.line(img, (x,y), (x, y + l), (255, 0, 255), thickness= t)
        # Top right
        cv2.line(img, (x1,y), (x1-l, y), (255, 0, 255), thickness= t)
        cv2.line(img, (x1,y), (x1, y + l), (255, 0, 255), thickness= t)
        
        #bottom left
        cv2.line(img, (x,y1), (x+l, y1), (255, 0, 255), thickness= t)
        cv2.line(img, (x,y1), (x, y1 - l), (255, 0, 255), thickness= t)
        
        #bottom right
        cv2.line(img, (x1,y1), (x1-l, y1), (255, 0, 255), thickness= t)
        cv2.line(img, (x1,y1), (x1, y1 - l), (255, 0, 255), thickness= t)
        
        return img
def main():
    cap = cv2.VideoCapture(0)

    current_time = 0
    prev_time = 0
    detector = FaceDetector()
    while True:
        ret, img = cap.read()
        img, bboxs = detector.findFaces(img)
        current_time = time.time()
        fps = 1/ (current_time - prev_time)
        prev_time = current_time
        cv2.putText(img, 'fps = ' + str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
if __name__ == '__main__':
    main()