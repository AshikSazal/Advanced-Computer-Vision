import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("AiTrainer/curls.mp4")

detector = pm.PoseDetector()
count = 0
direction = 0
previousTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img,(1280,720))
    # img = cv2.imread("AiTrainer/test.jpg")
    # img = cv2.resize(img,(720,480))
    img = detector.finedPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList)!=0:
        # # Right Arm
        # detector.findAngle(img, 12, 14, 16)
        # Left Arm
        angle = detector.findAngle(img, 11, 13, 15)
        percentage = np.interp(angle, (210,310),(0,100))
        bar = np.interp(angle, (220,310),(650,100))

        # check for the dumbbell curls
        color = (255,0,255)
        if percentage == 100:
            color = (0,255,0)
            # going up
            if direction == 0:
                count += 0.5
                direction = 1

        if percentage == 0:
            color = (0,255,0)
            # going down
            if direction == 1:
                count += 0.5
                direction = 0

        # Draw Bar
        cv2.rectangle(img, (1100,100), (1175,650), color,3)
        cv2.rectangle(img, (1100,int(bar)), (1175,650), color,cv2.FILLED)
        cv2.putText(img, f'{int(percentage)} %', (1100,75), cv2.FONT_HERSHEY_PLAIN, 3, color,3)

        # Draw Curl count
        cv2.rectangle(img, (0,450), (250, 720), (0,255,0),cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255,0,0),25)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0),5)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break