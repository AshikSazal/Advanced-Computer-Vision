import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Create model
mpHands = mp.solutions.hands
hands = mpHands.Hands(2,1,1)
mpDraw = mp.solutions.drawing_utils

# time
previousTime = 0
currentTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        # for multiple hand
        for handLandmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(handLandmarks.landmark):
                # print(id,landmark)
                height, width, channel = img.shape
                centerX, centerY = int(landmark.x * width), int(landmark.y * height)
                if id == 0:
                    cv2.circle(img, (centerX, centerY), 25, (255, 0, 255), cv2.FILLED)
            # draw of handmark and draw connection
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    # cv2.putText(image, time, position, font, scale, color, thickness)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break