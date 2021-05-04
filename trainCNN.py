import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import keras.layers

if __name__=='__main__':
    x_files = glob.glob('x*.txt')
    y_files = glob.glob('y*.txt')

    x = np.empty((1, 70, 210, 3))
    y = np.empty(1)

    for x_f, y_f in zip(x_files, y_files):
        xdata = np.loadtxt(x_f, delimiter=',')
        x = np.append(x, xdata.reshape(-1, 70, 210, 3), axis=0)
        y = np.append(y, np.loadtxt(y_f, delimiter=','), axis=0)
    x = np.delete(x, 0, 0)
    y = np.delete(y, 0, 0)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = Sequential([
        keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=(70, 210, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 7, activation="relu", padding="same"),
        keras.layers.Conv2D(128, 7, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test))
    result = model.evaluate(X_test, Y_test, batch_size=64)
    print(result)

    model.save('drowsiness_CNNmodel.h5')