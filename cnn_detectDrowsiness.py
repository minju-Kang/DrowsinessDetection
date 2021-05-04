import cv2
import dlib
from tensorflow.python.keras.models import load_model
import numpy as np

stream = cv2.VideoCapture(0)    # webcam
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = load_model('drowsiness_CNNmodel.h5')
closed = 0

while(stream.isOpened()):
    ret, frame = stream.read()

    if ret:
        img = cv2.flip(frame, 1)
        dets = detector(img, 1)  # detect driver's face

        for face in dets:
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)

            landmarks = predictor(img, face).parts()

            leye = img[landmarks[37].y:landmarks[41].y + 1, landmarks[36].x:landmarks[39].x + 1]
            reye = img[landmarks[44].y:landmarks[46].y + 1, landmarks[42].x:landmarks[45].x + 1]

            scaled_leye = cv2.resize(leye, (210, 70), interpolation=cv2.INTER_CUBIC)/255
            scaled_reye = cv2.resize(reye, (210, 70), interpolation=cv2.INTER_CUBIC)/255

            lyhat = round(model.predict(scaled_leye.reshape((1,70,210,3)))[0,0])
            ryhat = round(model.predict(scaled_reye.reshape((1,70,210,3)))[0,0])

            if not (lyhat | ryhat):
                closed += 1
                if closed > 3:
                    cv2.putText(img, "WakeUp!!", (face.left(), face.bottom() + 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            else:
                closed = 0

        cv2.imshow("Drowsy Driver Detection", img)

    if cv2.waitKey(1) == ord('q'):
        break
