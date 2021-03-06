import os
from os.path import join as pjoin
from random import Random, shuffle
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from keras.utils import to_categorical, plot_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras import regularizers

# highest accuracy: 38%
def main():
	os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
	try:
		data = scipy.io.loadmat('../images/GestureData.mat')['B']
		data = data.transpose(2, 1, 0)
		training_data, testing_data = create_test_data(data)
	except FileNotFoundError:
		training_data = scipy.io.loadmat('../images/TrainingData.mat')['D']
		testing_data = scipy.io.loadmat('../images/TestingData.mat')['B']

		training_data = training_data.transpose(2, 1, 0)
		testing_data = testing_data.transpose(2, 1, 0)

	training_labels, testing_labels = create_labels(training_data, testing_data)

	build_cnn(training_data, training_labels, testing_data, testing_labels)

	return

def build_cnn(X_train, y_train, X_test=None, y_test=None):

	X_train = X_train.reshape(len(X_train),1000,64,1)
	X_test = X_test.reshape(len(X_test),1000,64,1)

	# one-hot encode target column
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	#create model
	model = Sequential()
	#add model layers
	model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(1000,64,1)))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(6, activation='softmax', kernel_regularizer=regularizers.l2(0.003)))

	#compile model using accuracy to measure model performance
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	#train the model
	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4)

	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
	
	# model.predict(X_test[:4])

def create_labels(training_data, testing_data):
	training_labels = np.zeros(len(training_data))
	testing_labels = np.zeros(len(testing_data))

	inc_training = len(training_labels) / 6
	inc_testing = len(testing_labels) / 6

	class_label = -1
	for i in range(len(training_labels)):
		if i % inc_training == 0:
			class_label += 1
		training_labels[i] = class_label

	class_label = -1
	for i in range(len(testing_labels)):
		if i % inc_testing == 0:
			class_label += 1
		testing_labels[i] = class_label

	return training_labels, testing_labels

def create_test_data(test_set):
	validation_idx = []

	for i in range(len(test_set)):
		if i % 70 == 0:
			validation_idx.extend(range(i, i+10))

	validation_data = []

	for idx in validation_idx:
		validation_data.append(test_set[idx])

	validation_data = np.array(validation_data)
	training_data = np.delete(test_set, validation_idx, 0)

	print('training data has %d, validation data has %d' % (len(training_data), len(validation_data)))

	return training_data, validation_data

if __name__ == "__main__":
	main()