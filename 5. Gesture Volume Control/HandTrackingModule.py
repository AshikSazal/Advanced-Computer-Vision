import cv2
import mediapipe as mp
import time

# Create model
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, modelComplexity=1, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.modelComplex = modelComplexity
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            # for multiple hand
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                # draw of handmark and draw connection
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return img                   

    def findPosition(self,img,handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                # print(id,landmark)
                height, width, channel = img.shape
                centerX, centerY = int(landmark.x * width), int(landmark.y * height)
                lmList.append([id, centerX, centerY])
                if draw:
                    cv2.circle(img, (centerX, centerY), 10, (0, 255, 0), cv2.FILLED)

        return lmList



def main():
    # time
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime

        # cv2.putText(image, time, position, font, scale, color, thickness)
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()