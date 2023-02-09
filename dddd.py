import cv2
import cvlib as cv

image = cv2.imread('sss.jfif')
faces, confidences = cv.detect_face(image)

for (x,y,x2,y2), conf in zip(faces, confidences):
#확률 나타내기
    cv2.putText(image, str(conf), (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),1)
    cv2.rectangle(image, (x,y), (x2,y2), (0,255,0), 2)
    
cv2.imshow('image',image)
key = cv2.waitKey(0)
cv2.destroyAllWindows()