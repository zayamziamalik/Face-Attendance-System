import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'AttendanceImages'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)

for fln in myList:
    curimg = cv2.imread(f'{path}/{fln}')
    images.append(curimg)
    classNames.append(os.path.splitext(fln)[0])
# print(classNames)


def findEncodings(images):
    encodedlist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded = face_recognition.face_encodings(img)[0]
        encodedlist.append(encoded)
    return encodedlist

def markAttendance(name):
    now = datetime.now()
    dStr = now.strftime("%Y%m%d")
    filename = f"Attendance{dStr}.csv"
    if not os.path.exists(filename):
        with open(filename, "w") as file:
            file.write("Name, Time\n")
    with open(filename, "r+") as file:
        DataInFile=file.readlines()
        nameList = []
        for line in DataInFile:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dStr= now.strftime("%H:%M:%S")
            file.writelines(f"\n{name}, {dStr}")

knownEncodedList = findEncodings(images)
print('Encoding Complete')

capt= cv2.VideoCapture(0)

while True:
    success, img = capt.read()
    imgS=cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS=cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceloccurr = face_recognition.face_locations(imgS)
    encodedcurr = face_recognition.face_encodings(imgS, faceloccurr)

    for encodeFace, faceloc in zip(encodedcurr, faceloccurr):
        matches=face_recognition.compare_faces(knownEncodedList, encodeFace)
        faceDis=face_recognition.face_distance(knownEncodedList, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1= faceloc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255), 2)
            markAttendance(name)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)



# imageZzmTrn = face_recognition.load_image_file("ImagesBasic/zzmtrain.jpeg")
# imageZzmTrn = cv2.cvtColor(imageZzmTrn,cv2.COLOR_BGR2RGB)
# imageZzmTst = face_recognition.load_image_file("ImagesBasic/zzmtest1.jpeg")
# imageZzmTst = cv2.cvtColor(imageZzmTst,cv2.COLOR_BGR2RGB)