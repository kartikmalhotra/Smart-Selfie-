from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import dlib
import cv2
shape_predictor= "D:/facial/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
print(detector)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
def smile(mouth):
 A = dist.euclidean(mouth[3], mouth[9])
 B = dist.euclidean(mouth[2], mouth[10])
 C = dist.euclidean(mouth[4], mouth[8])
 L = (A+B+C)/3
 D = dist.euclidean(mouth[0], mouth[6])
 mar=L/D
 return mar


vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
c=0
t=0


while True:
 frame = vs.read()
 frame = imutils.resize(frame, width=450)
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 rects = detector(gray, 0)
 for rect in rects:
  shape = predictor(gray, rect)
  #print(shape)
  shape = face_utils.shape_to_np(shape)
  mouth= shape[mStart:mEnd]
  mar= smile(mouth)
  print(mar)
  mouthHull = cv2.convexHull(mouth)
  cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
 cv2.putText(frame, "MAR: {}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 if(mar > .4) :
     c=c+1
     if (c >= 15):
         t=t+1
         frame = vs.read()
         time.sleep(0.3)
         img_name = "opencv_frame_{}.png".format(t)
         #cv2.imwrite(img_name, frame)
         cv2.imshow("img", frame)
         print("{} written!".format(img_name))
         #cv2.destroyWindow("test") 
         c=0
         key2 = cv2.waitKey(0) & 0xFF
         if key2 == ord('q'):
             break
cv2.destroyAllWindows()
vs.stop()         