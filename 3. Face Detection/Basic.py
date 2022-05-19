import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/8.mp4")
previousTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
# 0.75 for false positive
faceDetection = mpFaceDetection.FaceDetection(0.75)

def rescale_frame(frame, percent=75):
    width = 1200
    height = 700
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

while True:
    success, img = cap.read()
    img = rescale_frame(img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            boundingBoxC = detection.location_data.relative_bounding_box
            height, width, channel = img.shape
            # bBox for drawing a rectangle cos we are not using mpDraw
            bBox = int(boundingBoxC.xmin * width), int(boundingBoxC.ymin * height),\
                int(boundingBoxC.width * width), int(boundingBoxC.height * height)
            print(type(bBox))
            cv2.rectangle(img,bBox,(255,0,255),2)
            cv2.putText(img, f'FPS: {str(int(detection.score[0]*100))}%', (bBox[0],bBox[1]-20), 
            cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255),2)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break