import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionConf=0.5):
        self.minDetectionConf = minDetectionConf

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        # 0.75 for false positive
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConf)


    def rescale_frame(self, frame, percent=75):
        width = 1200
        height = 700
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


    def findFaces(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection)
                boundingBoxC = detection.location_data.relative_bounding_box
                height, width, channel = img.shape
                # bBox for drawing a rectangle cos we are not using mpDraw
                bBox = int(boundingBoxC.xmin * width), int(boundingBoxC.ymin * height),\
                    int(boundingBoxC.width * width), int(boundingBoxC.height * height)
                bboxs.append([id, bBox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bBox)
                    cv2.putText(img, f'FPS: {str(int(detection.score[0]*100))}%', (bBox[0],bBox[1]-20), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255),2)

        return img, bboxs

    def fancyDraw(self,img,bbox, length=30, thickness =5, rectangleThickness=1):
        x, y, width, height = bbox
        # diagonal point or bottom right point at the diagonal position
        x1,y1 =x+width, y+height
        cv2.rectangle(img,bbox,(255,0,255),rectangleThickness)
        # Top left x,y
        cv2.line(img, (x,y),(x+length, y),(255,0,255),thickness)
        cv2.line(img, (x,y),(x, y+length),(255,0,255),thickness)

        # Top right x1,y
        cv2.line(img, (x1,y),(x1-length, y),(255,0,255),thickness)
        cv2.line(img, (x1,y),(x1, y+length),(255,0,255),thickness)

        # Bottom left x,y1
        cv2.line(img, (x,y1),(x+length, y1),(255,0,255),thickness)
        cv2.line(img, (x,y1),(x, y1-length),(255,0,255),thickness)

        # Bottom right x1,y1
        cv2.line(img, (x1,y1),(x1-length, y1),(255,0,255),thickness)
        cv2.line(img, (x1,y1),(x1, y1-length),(255,0,255),thickness)

        return img

def main():
    cap = cv2.VideoCapture("Videos/2.mp4")
    previousTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        if img is None:
            break
        img = detector.rescale_frame(img, percent=100)
        img, bboxs = detector.findFaces(img)

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime

        cv2.putText(img, f'FPS: {str(int(fps))}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()