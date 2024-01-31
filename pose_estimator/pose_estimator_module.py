import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode= False, model_complexity=1, upper_body = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.upper_body = upper_body
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.upper_body, self.smooth, self.detectionCon, self.trackCon)


    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img
    
    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy]) 
                cv2.circle(img, (cx,cy), 15, (255,0,255), -1)
        return lmList
    

    
    
    



def main():

    cap = cv2.VideoCapture(0)
    current_time = 0
    prev_time = 0
    detector = poseDetector()

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
    
if __name__ == '__main__':
    main()