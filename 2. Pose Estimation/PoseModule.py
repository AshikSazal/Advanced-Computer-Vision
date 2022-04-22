import cv2
import mediapipe as mp
import time

# create model
class PoseDetector():
    def __init__(self, mode=False, smooth=True, detectionConfidence=0.5,trackconfidence=0.5):
        self.mode = mode
        self.Complexity=1
        self.smooth = smooth
        self.enableSegmentation=False
        self.smoothSegmentation=True
        self.detectionConfidence = detectionConfidence
        self.trackconfidence = trackconfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.Complexity,self.smooth,self.enableSegmentation,self.smoothSegmentation,
        self.detectionConfidence,self.trackconfidence)

    def finedPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return img



    def findPosition(self, img, draw=True):
        lmList=[]
        # work on every land mark
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                # get image ratio from landmark and convert it to actual pixel value
                cx, cy = int(landmark.x*width), int(landmark.y*height)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

        return lmList

    def rescale_frame(self, frame, percent=75):
        width = 1200
        height = 700
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    
    

def main():
    # read video
    cap = cv2.VideoCapture('PoseVideos/12.mp4')
    previousTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.rescale_frame(img, percent=100)
        img = detector.finedPose(img)
        lmList = detector.findPosition(img)
        print(lmList)
        # # print specific value
        # lmList = detector.findPosition(img,draw=False)
        # if len(lmList) != 0:
        #     print(lmList[14][1],lmList[14][2])
        #     cv2.circle(img, (lmList[14][1],lmList[14][2]), 5, (255,0,0), cv2.FILLED)

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()