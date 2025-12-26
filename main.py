import cv2 # image processing library 
import numpy # scientific and numerical computing
import dlib # face recognition
from math import hypot


REC_COLOR = (0, 255, 0,)
REC_THIC = 1
ESC = 27 #ascii for escape on keyboard

L_EYE_MARKS = [36, 37, 38, 39, 40, 41]
R_EYE_MARKS = [42, 43, 44, 45, 46, 47]
BLINK_VAL = 7

FONT = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

cam = cv2.VideoCapture(0) #web camera capture
detector = dlib.get_frontal_face_detector() #detect face
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #collect eye location using .dat file

def midpoint(p1,p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2) # return the exact mid between two, can't be dec

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y) # detect left end of eye
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y) # detect right end of eye

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2])) # detect mid of top part of eye
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4])) # detect mid of bottom part of eye

    #    to sew corssed lines on camera
    #hor_line = cv2.line(frame, left_point, right_point, REC_COLOR,  REC_THIC) # draw a line between ends of eye
    #ver_line = cv2.line(frame, center_top, center_bottom, REC_COLOR,  REC_THIC) # draw a line between ends of eye

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1])) # cal length between sided fo eye
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1])) # cal length between bottom and top

    return hor_line_lenght / ver_line_lenght # ratio, blinking value comperison


while True:
    _, frame = cam.read() # _ indicate that a value returned by a function is being intentionally ignored.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray sacle saves computing power

    faces = detector(gray) #arry of faces detects location of face in 2d space
    for face in faces:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), REC_COLOR, REC_THIC) # surround face

        #blink detection
        landmarks = predictor(gray, face) # specific marks on face procesed
        left_eye_ratio = get_blinking_ratio(L_EYE_MARKS, landmarks)
        right_eye_ratio = get_blinking_ratio(R_EYE_MARKS, landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        print(blinking_ratio)

        if blinking_ratio > BLINK_VAL: 
            cv2.putText(frame, "BLINK", (50, 150), FONT, REC_THIC, REC_COLOR) #show indication when blinking
        
        #gaze
        l_eye_region = numpy.array([(landmarks.part(L_EYE_MARKS[0]).x,landmarks.part(L_EYE_MARKS[0]).y),
                                    (landmarks.part(L_EYE_MARKS[1]).x,landmarks.part(L_EYE_MARKS[1]).y),
                                    (landmarks.part(L_EYE_MARKS[2]).x,landmarks.part(L_EYE_MARKS[2]).y),
                                    (landmarks.part(L_EYE_MARKS[3]).x,landmarks.part(L_EYE_MARKS[3]).y),
                                    (landmarks.part(L_EYE_MARKS[4]).x,landmarks.part(L_EYE_MARKS[4]).y),
                                    (landmarks.part(L_EYE_MARKS[5]).x,landmarks.part(L_EYE_MARKS[5]).y)], numpy.int32) # define points to connect 
        #cv2.polylines(frame, [l_eye_region], True, (0 ,0 ,255), REC_THIC) # draw the eye frame

        height, width, _ = frame.shape
        mask = numpy.zeros((height, width), numpy.uint8) # mask only the eye to be white
        

        cv2.polylines(mask, [l_eye_region], True, 255, 2) # draw the eye frame
        cv2.fillPoly(mask, [l_eye_region], 255)

        min_x = numpy.min(l_eye_region[:, 0])
        max_x = numpy.max(l_eye_region[:, 0])
        min_y = numpy.min(l_eye_region[:, 1])
        max_y = numpy.max(l_eye_region[:, 1]) # eye borders

        eye = frame[min_y : max_y, min_x: max_x] # cut only eye to frame
        eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) # gray scale
        _, threshold_eye = cv2.threshold(eye_gray, 70, 255, cv2.THRESH_BINARY)

        cv2.imshow("EYE", cv2.resize(eye, None, fx=10, fy=10)) # window of only eye
        cv2.imshow("THRESHOLD", cv2.resize(threshold_eye, None, fx=10, fy=10)) # window of only eye gray scale

        cv2.imshow("MASK", mask) # cover all in black

    #show camera
    mirrored_frame = cv2.flip(frame, 1) # Flip the frame horizontally (mirror effect) 
    cv2.imshow("Camera", mirrored_frame) # will popup a window named "Camera" 
 
    #end sesssion
    key = cv2.waitKey(1) # if 'ESC' on keyboard is pressed, exit web cam
    if key == ESC:
        break

    elif cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1: # Check if the window was closed via the "X" button
        break

cam.release()
cv2.destroyAllWindows() 