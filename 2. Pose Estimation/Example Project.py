import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('PoseVideos/1.mp4')
previousTime = 0
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    img = detector.rescale_frame(img, percent=100)
    img = detector.finedPose(img)
    lmList = detector.findPosition(img)
    print(lmList)
    # # print specific value
    # lmList = detector.findPosition(img,draw=False)
    # print(lmList[14][1],lmList[14][2])
    # cv2.circle(img, (lmList[14][1],lmList[14][2]), 5, (255,0,0), cv2.FILLED)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break