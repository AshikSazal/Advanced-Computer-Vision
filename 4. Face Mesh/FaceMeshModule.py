import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,staticMode=False,maxFaces=2,landmarks=False,minCon=0.5,minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.landmarks = landmarks
        self.minCon = minCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.landmarks,
                                                    self.minCon,self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)


    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLandmarks in self.results.multi_face_landmarks:
                face = []
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_CONTOURS, 
                    self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),self.drawSpec)
                # get actual points
                for id, landmark in enumerate(faceLandmarks.landmark):
                    iHeight, iWidth, iChannel = img.shape
                    x, y = int(landmark.x*iWidth), int(landmark.y*iHeight)
                    cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                    # print(id,x,y)
                    face.append([x,y])
                    print(len(face))
                faces.append(face)

        return img, faces


    def rescale_frame(self, frame, percent=75):
        width = 1200
        height = 700
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)





def main():
    cap = cv2.VideoCapture("Videos/8.mp4")
    previousTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        if img is None:
            break
        img = detector.rescale_frame(img)
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))
        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img, f'FPS : {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        cv2.imshow("Image",img)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()