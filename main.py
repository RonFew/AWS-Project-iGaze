import cv2 # image processing library 
import numpy # scientific and numerical computing
import dlib # face recognition
from math import hypot


REC_COLOR = (0, 255, 0,)
REC_THIC = 1
ESC = 27 #ascii for escape on keyboard
LEFT_EYE = [37, 38, 40, 41]

cam = cv2.VideoCapture(0) #web camera capture
detector = dlib.get_frontal_face_detector() #detect face
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #collect eye location using .dat file

def midpoint(p1,p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2) # return the exact mid between two, can't be dec


while True:
    _, frame = cam.read() # _ indicate that a value returned by a function is being intentionally ignored.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray sacle saves computing power

    faces = detector(gray) #arry of faces detects location of face in 2d space
    for face in faces:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), REC_COLOR, REC_THIC) # surround face

        landmarks = predictor(gray, face)
        left_point = (landmarks.part(36).x, landmarks.part(36).y) # detect left end of eye
        right_point = (landmarks.part(39).x, landmarks.part(39).y)  # detect right end of eye
        top_point = midpoint(landmarks.part(37), landmarks.part(38)) # detect mid of top part of eye
        bottom_point = midpoint(landmarks.part(41), landmarks.part(40)) # detect mid of bottom part of eye

        hor_line = cv2.line(frame, left_point, right_point, REC_COLOR,  REC_THIC) # draw a line between ends of eye
        ver_line = cv2.line(frame, top_point, bottom_point, REC_COLOR,  REC_THIC) # draw a line between ends of eye

        hor_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1])) # cal length between sided fo eye
        ver_lenght = hypot((top_point[0] - bottom_point[0]), (top_point[1] - bottom_point[1])) # cal length between bottom and top pt
        print(hor_lenght/ver_lenght)

        #ratio = (hor_lenght/ver_lenght)
        #if ratio > 6:
        #    cv2.putText(frame, "BLINK", (50, 150), font, REC_THIC, REC_COLOR)

    cv2.imshow("Camera", frame) # will popup a window named "Camera" 
 
    key = cv2.waitKey(1) # if 'ESC' on keyboard is pressed, exit web cam
    if key  == ESC:
        break

cam.release()
cv2.destroyAllWindows() 