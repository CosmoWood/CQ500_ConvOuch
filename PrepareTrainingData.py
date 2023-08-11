# This code will prepare the training data before running this,
# we first need to run the reslicing notebook to reslice everything
# to 5mm then in this notebook we first exclude everything that is
# below 28 or beyond 50 scans then for each input scan, get all of
# the images and prepare a label for them generate 1 numpy array
# for each slice in each scan

#read nii files
import os
import numpy as np
import nibabel as nib

import pandas as pd
from matplotlib import pyplot as plt
from skimage.transform import resize

label_file_name = 'summary_label.csv'
nifti_folder = 'Nifti_Files'
new_size = (224, 224)
#the following 2 parameters can not be 1 at the same time
normalize_img = 0
#if 1, normalize to 0~255
#if 0, don't normalize, use raw numbers
three_channels = 1
#if 1, create 3 channels instead of the original 1 channel
#if 0, use the original 1 channle
#for the 3 channels, we use the following window:
#brain window: (l = 40, w = 80), i.e. 0~80 (everything <0 is 0, everything >80 is 255)
#bone window: (l = 500, w = 3000), i.e. -1000~2000
#subdural window: (l = 175, w = 50), i.e. 150~200
#l is the center of the luminance and w is the width
channel_param = ((0, 80), (-1000, 2000), (150, 200))

#we will only include data that is within this range after reslicing to 5mm
max_num_slice = 50
min_num_slice = 28


def create_channel(img, channel_min, channel_max):
    # this function will create channeled data based on the original single values data

    img_channel = img.copy()
    img_channel[img_channel <= channel_min] = 0
    img_channel[img_channel >= channel_max] = channel_max
    img_channel = 255 * (img_channel - np.amin(img_channel)) / (np.amax(img_channel) - np.amin(img_channel))
    return img_channel


# we need to count how many slices are in each nifti file, probably go with the smallest number
# then for each scan, get the middle slices with that number
# after preprocessing, we don't have scans from 120, 163, 336 and 491
scan_list = np.array([range(1, 120)])
scan_list = np.concatenate((scan_list, np.array([range(121, 163)])), axis=1)
scan_list = np.concatenate((scan_list, np.array([range(164, 336)])), axis=1)
scan_list = np.concatenate((scan_list, np.array([range(337, 491)])), axis=1)

num_slice = np.empty((scan_list.shape[1],))

j = 0
for ss in scan_list[0]:
    # for each scan read in the nifti file
    i = ss
    scan_fn = nifti_folder + '/CQ500-CT-' + str(i) + '_5mm.nii'
    img = nib.load(scan_fn)
    img_data = img.get_fdata()
    num_slice[j] = img_data.shape[2]
    j = j + 1

num_slice = num_slice.astype(int)

scan_list = np.squeeze(scan_list)
# create a pandas dataframe with the index and num_slice
num_slice_df = pd.DataFrame({'scan_id': scan_list, 'num_slice': num_slice})

num_slice_df = num_slice_df[num_slice_df.num_slice <= max_num_slice]
num_slice_df = num_slice_df[num_slice_df.num_slice >= min_num_slice]

# num_slice_df has all of the information about number of slices in each scan



# now we will read in each scan, select min_num_slices from the middle of the scan, and
# generate a numpy file for each slice, with the corresponding label!

final_scan_list = num_slice_df['scan_id']

labels = pd.read_csv(label_file_name)

# use the scan name as index
labels.set_index('name', inplace=True)

for ss in final_scan_list:
    # for each scan read in the nifti file
    #scan_fn = nifti_folder + '/CQ500-CT-' + str(ss) + '/CQ500-CT-' + str(ss) + '_5mm.nii'
    scan_fn = nifti_folder + '/CQ500-CT-' + str(i) + '_5mm.nii'
    img = nib.load(scan_fn)
    img_data = img.get_fdata()

    # check the number of slices for this scan, and select the middle 28
    current_num_slice = img_data.shape[2]

    if current_num_slice > min_num_slice:
        # more than 28 slices, we need to select the middle one
        if current_num_slice % 2 == 2:
            # this is an even number
            start_slice = int((current_num_slice - min_num_slice) / 2)

        else:
            # this is an odd number
            start_slice = int((current_num_slice - 1 - min_num_slice) / 2)

        # in python, the end_slice is actually not included in the range
        end_slice = int(start_slice + min_num_slice)
    else:
        start_slice = 0
        end_slice = int(start_slice + min_num_slice)

    # grab the label for this scan
    current_label = labels['ICH']['CQ500-CT-' + str(ss)]
    current_scan_id = ss

    # now we generate data for slices in the range between start_slice and end_slice
    slice_index = 0  # this starts from 0, not the same as the original slice number in the original scan
    for j in range(start_slice, end_slice):
        # resize
        one_img_resize = resize(img_data[:, :, j], new_size)
        if normalize_img == 1:
            # normalize to 0~255
            img_processed = 255 * (one_img_resize - np.amin(one_img_resize)) / (
                        np.amax(one_img_resize) - np.amin(one_img_resize))
        else:
            if three_channels == 1:
                # be careful! now each image will be 3D

                # first create an empty numpy array for the channeled ones
                img_processed = np.empty((new_size[0], new_size[1], len(channel_param)))

                for c in range(0, len(channel_param)):
                    img_processed[:, :, c] = create_channel(one_img_resize, channel_param[c][0], channel_param[c][1])

            else:
                img_processed = one_img_resize
        # save this img_processed as a numpy array
        img_file_name = 'Slices/' + 'CQ500-CT-' + str(ss) + '_Slice' + str(slice_index) + '.npy'
        np.save(img_file_name, img_processed)

        # create a dictionary for label and scan_id and save it
        label_info = {'scan_id': current_scan_id, 'label': current_label}
        img_label_file_name = 'Labels/' + 'CQ500-CT-' + str(ss) + '_Slice' + str(slice_index) + '.npy'
        np.save(img_label_file_name, label_info)

        print(label_info)

        slice_index = slice_index + 1