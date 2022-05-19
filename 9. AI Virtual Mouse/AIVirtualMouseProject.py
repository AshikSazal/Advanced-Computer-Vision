import cv2
import numpy as np
import time
import HandTrackingModule as htm
import autopy

widthCam, heightCam = 640, 480
frameReduction = 100 # Frame Reduction
smoothening = 7
previousLocationX, previousLocationY = 0, 0
currectLocationX, currentLocationY = 0, 0

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
previousTime = 0
detector = htm.handDetector(maxHands=1)
# get the width and height of the screen individual screen
widthScreen, heightScreen = autopy.screen.size()

while True:
    success, img = cap.read()

    #  1. Find hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        cv2.rectangle(img, (frameReduction, frameReduction),(widthCam-frameReduction,heightCam-frameReduction),
        (255,0,255),2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameReduction, widthCam-frameReduction),(0, widthScreen))
            y3 = np.interp(y1, (frameReduction, heightCam-frameReduction),(0, heightScreen))

            # 6. Smoothen Values by reduce the value (Mouse work in smooth)
            currectLocationX = previousLocationX + (x3 - previousLocationX) / smoothening
            currectLocationY = previousLocationY + (y3 - previousLocationY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(widthScreen - currectLocationX, currectLocationY)# widthScreen-x3 flip screen
            cv2.circle(img, (x1,y1), 15, (255,0,255),cv2.FILLED)
            previousLocationX, previousLocationY = currectLocationX, currectLocationY

        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:

            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # 10. Click mouse if distance short
            if length < 30:
                cv2.circle(img, (lineInfo[4],lineInfo[5]), 15, (0,255,0),cv2.FILLED)
                autopy.mouse.click() # Mouse click add two fingers

    # 11. Frame Rate
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)

    # 12. Display

    cv2.imshow("image",img)
    if cv2.waitKey(1) == ord('q'):
        break