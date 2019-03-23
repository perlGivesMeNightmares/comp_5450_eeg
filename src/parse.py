import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from os import path


def parse_data():
	basepath = path.dirname(__file__)
	fp = path.abspath(path.join(basepath, "..", "data", "01", "1_raw_data_13-12_22.03.16.txt"))
	return np.loadtxt(fp, delimiter="\t", skiprows=1)	

def mat_to_columns(d):
	column_dict = {}
	column_dict['time'] = d[:, 0]
	column_dict['gesture'] = d[:, 9]

	for col_idx in range(1, 9):
		column_dict['channel_'+str(col_idx)] = d[:, col_idx]

	return column_dict

def label_data(d, channel_num):
	channel_col = d['channel_' + str(channel_num)]
	label_col = d['gesture']
	combined = np.column_stack((channel_col, label_col))
	print('combined is %s' % len(combined))
	return combined

def make_stft(data):

	Pxx, freqs, bins, im = plt.specgram(data['channel_1'], NFFT=256, Fs=1000, noverlap=128, cmap=plt.cm.gist_heat)
	

def filter_by_gesture(data, gest_type):
	"""get the first contiguous block of this gesture type"""
	print(len(data))
	return
	filtered = data[data[:, 1] == gest_type]
	channel_data = filtered[:, 0]
	return channel_data


def main():
	CHANNEL = 1
	GESTURE = 1

	data = parse_data()
	columns = mat_to_columns(data)
	labeled = label_data(columns, CHANNEL)
	gesture_data = filter_by_gesture(columns, GESTURE)
	return
	graphs = make_stft(gesture_data)
	
	plt.show()
	return

if __name__ == "__main__":
	main()


# we cant use the whole vector
# plt.plot(time_col, channel_1)