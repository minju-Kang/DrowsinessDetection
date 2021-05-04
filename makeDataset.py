import cv2
import dlib
import numpy as np
import sys

def takeEyePic():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    stream = cv2.VideoCapture(0)    # webcam
    data = np.empty(shape=(1,70,210,3), dtype=np.int64)

    while(stream.isOpened()):
        ret, frame = stream.read()

        if ret:
            img = cv2.flip(frame, 1)
            dets = detector(img, 1)  # detect driver's face

            for face in dets:
                landmarks = predictor(img, face).parts()

                leye = img[landmarks[37].y:landmarks[41].y + 1, landmarks[36].x:landmarks[39].x + 1]
                reye = img[landmarks[44].y:landmarks[46].y + 1, landmarks[42].x:landmarks[45].x + 1]

                scaled_leye = cv2.resize(leye, (210,70), interpolation=cv2.INTER_CUBIC)
                scaled_reye = cv2.resize(reye, (210,70), interpolation=cv2.INTER_CUBIC)

                cv2.imshow("left eye", scaled_leye)

                data = np.append(data, [scaled_leye], axis=0)
                data = np.append(data, [scaled_reye], axis=0)

        if cv2.waitKey(1) == ord('q'):
            return np.delete(data, 0, 0)


def makeData():
    f_x = input("x file name:")
    f_y = input("y file name:")

    openedData = takeEyePic()
    cv2.waitKey(0)
    closedData = takeEyePic()
    x = np.append(openedData, closedData, axis=0)

    opened_y = np.ones(openedData.shape[0])
    closed_y = np.zeros(closedData.shape[0])
    y = np.append(opened_y, closed_y, axis=0)

    x = x.reshape(-1, 70 * 210 * 3) / 255
    np.savetxt(f_x, x, delimiter=',')
    np.savetxt(f_y, y, delimiter=',')

    return x,y


if __name__=='__main__':
    x, y = makeData()
    print(x.shape)
    print(y.shape)