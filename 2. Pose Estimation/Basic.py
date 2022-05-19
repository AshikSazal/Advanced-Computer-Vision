import cv2
import mediapipe as mp
import time

# create model
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# read video
cap = cv2.VideoCapture('PoseVideos/12.mp4')

previousTime = 0

def rescale_frame(frame, percent=75):
    width = 1200
    height = 700
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

while True:
    success, img = cap.read()
    img = rescale_frame(img, percent=100)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        # work on every land mark
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            # get image ratio from landmark and convert it to actual pixel value
            cx, cy = int(landmark.x*width), int(landmark.y*height)
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break