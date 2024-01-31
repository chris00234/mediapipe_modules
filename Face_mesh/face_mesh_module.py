import cv2
import mediapipe as mp
import time

class FaceMesh():
    def __init__(self, mode = False, maxFaces = 2, module_complexity = 1, minDetectionCon = 0.5, minTrackCon= 0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.module_complexity = module_complexity
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.mode, max_num_faces=self.maxFaces, min_detection_confidence=self.minDetectionCon, min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius= 1)

    def findFaceMesh(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, landmark_drawing_spec = self.drawSpec)

                face = []
                for id,lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    x,y = int(lm.x * w), int(lm.y * h)
                    cv2.putText(img, 'x', (x,y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 1)
                    face.append([x,y])
                faces.append(face)
        return img, faces
                    
    

    
def main():
    cap = cv2.VideoCapture(0)
    current_time = 0
    prev_time = 0
    detector = FaceMesh(maxFace = 2)
    while True:
        ret, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))
        #Showing frame
        current_time = time.time()
        fps = 1/(current_time - prev_time)
        prev_time = current_time
        cv2.putText(img, 'fps = ' + str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()