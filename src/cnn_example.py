from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

def test_gpu():
	from keras import backend as K
	K.tensorflow_backend._get_available_gpus()

def main():
	test_gpu()
	return
	#download mnist data and split into train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	#plot the first image in the dataset
	plt.imshow(X_train[0])
	plt.show()

	#check image shape
	print(X_train[0].shape)

	#reshape data to fit model (number of images, length, width, color_mode)
	X_train = X_train.reshape(60000,28,28,1) / 256
	X_test = X_test.reshape(10000,28,28,1) / 256

	# one-hot encode target column
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	print(y_train[0])

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
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
	model.predict(X_test[:4])

if __name__ == "__main__":
	main()