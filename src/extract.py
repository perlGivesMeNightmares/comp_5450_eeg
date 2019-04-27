import os
import numpy as np
import pprint

def parse_data():
	basepath = path.dirname(__file__)
	fp = path.abspath(path.join(basepath, "..", "data", "01", "1_raw_data_13-12_22.03.16.txt"))
	return np.loadtxt(fp, delimiter="\t", skiprows=1)	

def write_mat_data(data, folder_name, parent_filename, gesture):
	new_filename = 'gesture_' + str(gesture) + '_user_' + folder_name + '_' + parent_filename
	np.savetxt(new_filename, data, delimiter="\t", fmt="%s")
	return True

def filter_by_gesture(data, gesture):
	gest_col = data[:, 9]

	gesture_index_start = None
	gesture_index_end = None
	found_flag = False

	lst = []

	for idx, i in enumerate(gest_col):
		if found_flag and i == gesture:
			lst.append(data[idx])
		elif found_flag and i != gesture:
			gesture_index_end = idx
			break
		elif i == gesture:
			gesture_index_start = idx
			found_flag = True

	return lst

def main():
	gesture_types = [0]  # range(1, 8)

	basepath = os.path.dirname(__file__)
	fp = os.path.abspath(os.path.join(basepath, "..", "data"))
	subdirectories = [name for name in os.listdir(fp) if os.path.isdir(os.path.join(fp, name))]
	for dir_idx, directory in enumerate(subdirectories):
		# if dir_idx != 0:
		# 	continue
		sd_fp = os.path.abspath(os.path.join(basepath, "..", "data", str(directory)))
		subfiles = os.listdir(sd_fp)
		for file_name in subfiles:	
			fp = os.path.abspath(os.path.join(basepath, "..", "data", directory, file_name))
			txt_data = np.loadtxt(fp, delimiter="\t", skiprows=1)
			gest_dict = {}
			for gest in gesture_types:
				result = filter_by_gesture(txt_data, gest)
				if len(result) > 0:
					success = write_mat_data(result, directory, file_name, gest)		

	return

if __name__ == "__main__":
	main()


# we cant use the whole vector
# plt.plot(time_col, channel_1)