import cv2
import time
import os
import HandTrackingModule as htm

widthCam, heightCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
overlayList = []
for imagePath in myList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    overlayList.append(image)

previousTime = 0

detector = htm.handDetector(detectionConfidence=0.75)

fingerTipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)

    if len(landmarkList) != 0:
        fingers = []
        # Thumb
        # x=1 index value for x axis
        if landmarkList[fingerTipIds[0]][1] > landmarkList[fingerTipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers
        for id in range(1,5):
            # y=2 index value for y axis
            if landmarkList[fingerTipIds[id]][2] < landmarkList[fingerTipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)

        # totalFingers-1 for the index which start at 0
        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

        # Show the count value
        cv2.rectangle(img, (20,225), (170, 425), (0,255,0),cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0),25)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime=currentTime
    cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break