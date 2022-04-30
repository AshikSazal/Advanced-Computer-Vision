import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/2.mp4")
previousTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

def rescale_frame(frame, percent=75):
    width = 1200
    height = 700
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

while True:
    success, img = cap.read()
    if img is None:
        break
    img = rescale_frame(img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLandmarks, mpFaceMesh.FACEMESH_CONTOURS, 
            mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),drawSpec)
            # get actual points
            for id, landmark in enumerate(faceLandmarks.landmark):
                iHeight, iWidth, iChannel = img.shape
                x, y = int(landmark.x*iWidth), int(landmark.y*iHeight)
                print(id,x,y)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img, f'FPS : {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    cv2.imshow("Image",img)

    if cv2.waitKey(1) == ord('q'):
        break