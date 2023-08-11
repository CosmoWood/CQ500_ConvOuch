import glob
import pickle
import re
import torch
#from tensorflow import keras


# import numpy as np
# from keras import optimizers
# from keras.applications.vgg16 import VGG16
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import Flatten
# from keras.models import Model
#
# from CQ500DataGenerator import DataGenerator

# define our variables
data_dir = './'
num_slices_original = 28
num_slices_per_subject = 24       # always using 16 slices per subject
start_slice = (num_slices_original - num_slices_per_subject)/2
end_slice = start_slice + num_slices_per_subject

start_slice = int(start_slice)
end_slice = int(end_slice)

# create list of IDs from all slices
data_dir = './'
all_IDs = set()
all_Slices = glob.glob(r".\Slices\CQ500-CT-*")

print(torch.cuda.is_available())
for item in all_Slices:
    subj_match = re.match(r".*CQ500-CT-([0-9]+)_Slice[0-9]+\.npy", item)
    subj_id = subj_match.group(1)
    all_IDs.add(subj_id)

all_IDs = list(all_IDs)
all_IDs.sort()

print(all_IDs)
