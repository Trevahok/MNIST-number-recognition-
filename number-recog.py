import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D , MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.models import model_from_json

np.random.seed(123)
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],1,28,28)
x_test = x_test.reshape(x_test.shape[0],1,28,28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_test /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

model = Sequential()
# model.add(Convolution2D(32,( 3, 3), activation='relu', input_shape=(1,28,28),data_format='channels_first'))
# model.add(Convolution2D(32,( 3, 3), activation='relu',data_format='channels_first'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
model.add(Convolution2D(32,(5,5), activation='relu',input_shape=(1,28,28),data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, (5, 5), activation='relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=4,verbose=1)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
score = model.evaluate(x_test,y_test,verbose=0)
print('score: ',score)
