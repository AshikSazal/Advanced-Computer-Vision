import cv2
import numpy as np
import time
import HandTrackingModule as htm
import os

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)
brushThickness = 15
xprev, yprev = 0, 0
imageCanvas = np.zeros((720,1280,3),np.uint8) # 3 channel
eraserThickness = 50

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.handDetector(detectionConfidence=0.85)

while True:
    # 1. Import Image
    success, img = cap.read()
    # flip image
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!=0:
        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:] # Index finger
        x2, y2 = lmList[12][1:] # Middle finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If selection mode - Two finger are up
        if fingers[1] and fingers[2]:
            # whenever selection mode
            xprev, yprev = 0, 0

            print("selection mode")
            # checking for the click
            if y1 < 125:
                if 210 < x1 < 350:
                    header = overlayList[0]
                    # B, G, R
                    drawColor = (255, 0, 255) # purple
                elif 435 < x1 < 565:
                    header = overlayList[1]
                    drawColor = (246,130,50) # sky blue
                elif 655 < x1 < 795:
                    header = overlayList[2]
                    drawColor = (0, 255, 0) # green
                elif 890 < x1 <1080:
                    header = overlayList[3]
                    drawColor = (0, 0, 0) # Black color
            cv2.rectangle(img, (x1,y1-25), (x2, y2+25), drawColor, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            print("drawing mode")
            # starting draw from current position of finger
            if xprev==0 and yprev==0:
                xprev, yprev = x1, y1

            # for eraser selection which is black color
            if drawColor == (0,0,0):
                cv2.line(img, (xprev,yprev),(x1,y1),drawColor,eraserThickness)
                cv2.line(imageCanvas, (xprev,yprev),(x1,y1),drawColor,eraserThickness)

            # for other color selection
            else:
                # draw line
                cv2.line(img, (xprev,yprev),(x1,y1),drawColor,brushThickness)
                cv2.line(imageCanvas, (xprev,yprev),(x1,y1),drawColor,brushThickness)
            xprev, yprev = x1, y1

    imgGray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imageCanvas) # combine image original and canvas

    # Setting the header image
    img[0:125,0:1280]=header
    # draw in original image
    # img = cv2.addWeighted(img, 0.5, imageCanvas, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.imshow("Canvus", imageCanvas)
    if cv2.waitKey(1) == ord('q'):
        break