from keras.datasets import mnist
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical, plot_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

def test_gpu():
	from keras import backend as K
	K.tensorflow_backend._get_available_gpus()

def main():
	os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
	# test_gpu()
	#download mnist data and split into train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train[0:5000]
	y_train = y_train[0:5000]

	X_test = X_test[0:1000]
	y_test = y_test[0:1000]

	#reshape data to fit model (number of images, length, width, color_mode)
	X_train = X_train.reshape(5000,28,28,1) / 256
	X_test = X_test.reshape(1000,28,28,1) / 256

	print(X_train.shape)

	# one-hot encode target column
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	#create model
	model = Sequential()
	#add model layers
	model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
	model.add(Conv2D(32, kernel_size=3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(10, activation='softmax'))

	#compile model using accuracy to measure model performance
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	#train the model
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

	plot_model(model, to_file='./model.png')
	#model.save('./')


	# model.predict(X_test[:4])

if __name__ == "__main__":
	main()