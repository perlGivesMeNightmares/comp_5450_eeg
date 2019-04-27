import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np

classes = ['gesture_'+str(i) for i in range(1,8)]

def get_paths():
	basepath = os.path.dirname(__file__)
	fp = os.path.abspath(pjoin(basepath, "..", "images"))
	img_paths = [name for name in os.listdir(fp) if os.path.isfile(pjoin(fp, name)) and 'Image' in name]
	return [(os.path.abspath(pjoin(basepath, "..", "images", img)), img[:9]) for img in img_paths]

def main():
	images = []

	for idx, ipath in enumerate(get_paths()):

		if idx > 0:
			break

		print('Classifying %s' % str(ipath[1]))

		if "gesture_7" in ipath:
			continue

		# quirk of the dataset
		raw_img = np.genfromtxt(ipath[0], delimiter="\t")[:,0:-1]
		print('image dimensions are %d, %d' % (len(raw_img), len(raw_img[1])))
		images.append(raw_img)

	# build_cnn(images)

	return

def build_cnn(X_train, y_train, X_test, y_test):

	#plot the first image in the dataset
	plt.imshow(X_train[0])
	plt.show()

	#check image shape
	print(X_train[0].shape)

	#reshape data to fit model (number of images, length, width, color_mode)
	X_train = X_train.reshape(500,48,48,1) / 256
	X_test = X_test.reshape(40,48,48,1) / 256

	# one-hot encode target column
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	print(y_train[0])

	#create model
	model = Sequential()
	#add model layers
	model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(48,48,1)))
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