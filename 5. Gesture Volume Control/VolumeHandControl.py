import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
# import audio control file
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

widthCamera, heightCamera = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, widthCamera)
cap.set(4, heightCamera)
previousTime = 0
vol = 0
volBar = 400
volPercentage = 0

detector =htm.handDetector(detectionConfidence=0.7)

# For volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volumeRange = volume.GetVolumeRange()

minVol = volumeRange[0]
maxVol = volumeRange[1]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    if len(landmarkList)!=0:
        x1, y1 = landmarkList[4][1], landmarkList[4][2]
        x2, y2 = landmarkList[8][1], landmarkList[8][2]

        # center of line
        cx, cy = (x1+x2)//2, (y1+y2)//2


        cv2.circle(img, (x1,y1), 10, (255,0,255),cv2.FILLED)
        cv2.circle(img, (x2,y2), 10, (255,0,255),cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img, (cx,cy), 10, (255,0,255),cv2.FILLED)

        # length between thumb_tip and index_finger_tip
        length = math.hypot(x2-x1,y2-y1)
        # show the hand range (min,max)
        # print(length)
        
        # Volume range -65 - 0

        vol = np.interp(length, [15,107],[minVol, maxVol])
        volBar = np.interp(length, [15,107],[400, 150])
        volPercentage = np.interp(length, [15,107],[0, 100])
        # print to show hand range and volume range
        # print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)
        
        if length<50:
            cv2.circle(img, (cx,cy), 10, (0,255,0),cv2.FILLED)

    cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3)
    cv2.rectangle(img, (50,int(volBar)), (85,400), (0,255,0), cv2.FILLED)
    cv2.putText(img, f'{int(volPercentage)} %', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(img, f'FPS: {int(volPercentage)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break