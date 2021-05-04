import cv2
import dlib

def eyeRatio(landmarks):
    # calculate eye height
    l_height = landmarks[40].y - landmarks[38].y
    r_height = landmarks[47].y - landmarks[43].y

    # calculate eye width
    l_width = landmarks[39].x - landmarks[36].x
    r_width = landmarks[45].x - landmarks[42].x

    # calculate eye ratio
    l_ratio = l_height / l_width
    r_ratio = r_height / r_width
    avg_ratio = (l_ratio + r_ratio) / 2

    print(l_height)
    print(l_width)
    print(avg_ratio)

    return avg_ratio


stream = cv2.VideoCapture(0)    # webcam
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
closed = 0
threshold = 0.35

while(stream.isOpened()):
    ret, frame = stream.read()

    if ret:
        img = cv2.flip(frame, 1)
        dets = detector(img, 1)  # detect driver's face

        for face in dets:
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)
            print("Detection: Left: {} Top: {} Right: {} Bottom: {}".format(face.left(), face.top(), face.right(), face.bottom()))

            landmarks = predictor(img, face).parts()
            for p in landmarks:
                cv2.circle(img, (p.x, p.y), 2, (0, 255, 0), -1)

            if (eyeRatio(landmarks) < threshold):
                closed += 1
                if closed > 5:
                    cv2.putText(img, "WakeUp!!", (face.left(), face.bottom()+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            else:
                closed = 0
            cv2.imshow("Drowsy Driver Detection", img)

    if cv2.waitKey(1) == ord('q'):
        break
