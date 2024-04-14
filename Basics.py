import cv2
import numpy as np
import face_recognition

imageZzmTrn = face_recognition.load_image_file("ImagesBasic/zzmtrain.jpeg")
imageZzmTrn = cv2.cvtColor(imageZzmTrn,cv2.COLOR_BGR2RGB)
imageZzmTst = face_recognition.load_image_file("AttendanceImages/Abbas Ghori.jpeg")
imageZzmTst = cv2.cvtColor(imageZzmTst,cv2.COLOR_BGR2RGB)

faceloc =face_recognition.face_locations(imageZzmTrn)[0]
encodeZzm =face_recognition.face_encodings(imageZzmTrn)[0]
cv2.rectangle(imageZzmTrn, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255,0,255), 2)


facelocTst =face_recognition.face_locations(imageZzmTst)[0]
encodeZzmTst =face_recognition.face_encodings(imageZzmTst)[0]
cv2.rectangle(imageZzmTst, (facelocTst[3], facelocTst[0]), (facelocTst[1], facelocTst[2]), (255,0,255), 2)

results= face_recognition.compare_faces([encodeZzm], encodeZzmTst)
# print(faceloc)
print(results)



cv2.imshow('ZZM', imageZzmTrn)
cv2.imshow('ZZM TST', imageZzmTst)
cv2.waitKey(0)