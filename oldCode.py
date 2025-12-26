
#old code

cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), REC_COLOR, REC_THIC) # surround face
        landmarks = predictor(gray, face)
        x = landmarks.part(36).x
        y = landmarks.part(36).y
        cv2.circle(frame, (x, y), 3, REC_COLOR, REC_THIC)


'
def eye(pts):
    mid_pt = []
    for p1, p2 in zip(pts, pts[1:]):
        mid_pt.extend([int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)])
    return mid_pt

   hor_line = cv2.line(frame, eye(LEFT_EYE)[0], eye(LEFT_EYE)[1], REC_COLOR, REC_THIC)