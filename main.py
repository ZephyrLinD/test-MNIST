import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 由于mnist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # 归一化，所有数值在 0 - 1 之间
x_test /= 255
print(x_train.shape[0], 'train samples') # 60000
print(x_test.shape[0], 'test samples')   # 10000

batch_size = 128
num_classes = 10
epochs = 20


# convert class vectors to binary class matrices
print(y_train[0]) # 5
y_train = keras.utils.to_categorical(y_train, num_classes) # 把 y 变成了 one-hot 的形式
print(y_train[0]) # [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#model.summary()  # 打印出模型概况 

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, # verbose是屏显模式, 0是不屏显，1是显示一个进度条，2是每个epoch都显示一行数据
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])