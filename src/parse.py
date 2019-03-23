import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from os import path


def parse_channels():
	basepath = path.dirname(__file__)
	fp = path.abspath(path.join(basepath, "..", "data", "01", "1_raw_data_13-12_22.03.16.txt"))
	d = np.loadtxt(fp, delimiter="\t", skiprows=1)

	channels = {}
	time_col = d[:, 0]

	for col_idx in range(1, 9):
		channels['channel_'+str(col_idx)] = d[:, col_idx]

	return channels

def make_graphs(data):
	# teststft = spectrogram(data(:,1),hamming(256),50,512);

	Pxx, freqs, bins, im = plt.specgram(data['channel_1'], NFFT=256, Fs=1000, noverlap=128, cmap=plt.cm.gist_heat)

	# plt.plot(time_col, channel_1)


def main():
	data = parse_channels()
	graphs = make_graphs(data)
	
	plt.show()
	return

if __name__ == "__main__":
	main()


# we cant use the whole vector