import os
import glob
import re
import SimpleITK as sitk

import numpy as np
import pydicom
#import nibabel as nib
import dicom2nifti as d2n
data_dir = 'D:/dataset/CQ500/'

def RenameFolders():
    for dir_entry in os.scandir(path = data_dir):
        if dir_entry.is_dir():
            dir_re = re.search(pattern = "CQ500CT([0-9]*) CQ500CT([0-9]*)", string = dir_entry.name)
            dir_new = "CQ500-CT-%s" % dir_re.group(1)
            os.rename(data_dir + dir_entry.name, data_dir + dir_new)


# create a dictionary of all the sub directories for each subject directory
# loop through subject directories
#RenameFolders()

def dcm2nii(dcms_path, nii_path):
	# 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
	# 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
	# 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)

if __name__ == "__main__":
    subject_dir_dict = dict()
    for dir_entry in os.scandir(path = data_dir):
        #subject_dir_dict = dict()
        if dir_entry.is_dir() and re.match("CQ500-CT-([0-9]*)", dir_entry.name):
            subject_dir_dict[dir_entry.name] = list()

            # loop through sub-directories and exclude some by name
            for sub_dir in os.scandir(dir_entry.path + "/Unknown Study/"):
                if sub_dir.is_dir() and not re.search("(POST|(?<!PRE )CONTRAST|BONE| C | C$)", sub_dir.name):
                    dcm_files = glob.glob(sub_dir.path + "/*.dcm")
                    dicom_data = pydicom.read_file(dcm_files[0])

                    # add info for these sub-directories to a dictionary
                    sub_info = [sub_dir.name, dicom_data.PixelSpacing[0], dicom_data.PixelSpacing[1], dicom_data.SliceThickness, len(dcm_files)]
                    subject_dir_dict[dir_entry.name].append(sub_info)


            # warn if no usable directories found
            if(len(subject_dir_dict[dir_entry.name]) < 1):
                # print("Warning: {subject} found no usable subdirectories".format(subject=dir_entry.name))
                subject_dir_dict.pop(dir_entry.name)    # remove the entry from the dict
                continue

            elif(len(subject_dir_dict[dir_entry.name]) > 1):
                # parse the remaining directories to find the best candidate scan
                subject_scans = subject_dir_dict[dir_entry.name]

                scans_info = np.array(subject_scans)
                sorted_idx = np.argsort(scans_info[:, 3])

                # check for a 5mm slice scan
                if(float(scans_info[sorted_idx[-1], 3]) == 5.0):

                    # check if there are two 5mm slice scans
                    if(float(scans_info[sorted_idx[-2], 3]) == 5.0):
                        # if more than one 5mm slice scan, use the scan with fewest slices
                        scans_info = scans_info[scans_info[:, 3].astype(float) == 5.0]          # keep only the 5mm slice scans
                        sorted_slices_idx = np.argsort(scans_info[:, 4])
                        subject_dir_dict[dir_entry.name] = [subject_dir_dict[dir_entry.name][sorted_slices_idx[-1]]]  # replace with fewest slices study
                        if(subject_dir_dict[dir_entry.name][0][4] < 30):
                            print("Warning: Fewer than 30 slices for {}".format(dir_entry.path + "/" + subject_dir_dict[dir_entry.name][0]))
                    else:
                        subject_dir_dict[dir_entry.name] = [subject_dir_dict[dir_entry.name][sorted_idx[-1]]]         # replace with the 5mm study

                else:   # if no 5mm slice scans, so choose the smallest
                    subject_dir_dict[dir_entry.name] = [subject_dir_dict[dir_entry.name][sorted_idx[0]]]              # replace with the smallest slice scan


            # convert the chosen study folder to nifti for further processing
            #print(dir_entry.path + "/Unknown Study/" + subject_dir_dict[dir_entry.name][0][0])
            #loop through the dictionary
            #print(subject_dir_dict)
    output_dir = "./Nifti_Files"
    #d2n.convert_directory("D:/dataset/CQ500/CQ500-CT-100/Unknown Study/","./", True,False)
    # dcm2nii("D:/dataset/CQ500/CQ500-CT-2/Unknown Study/CT 5mm","D:/PyProjects/CQ500_ConvOuch/outputFolder")

    print(subject_dir_dict)
    for subject, study in subject_dir_dict.items():
        DicomData_dir = data_dir + subject + "/Unknown Study/" + study[0][0]
        print(DicomData_dir)
        d2n.convert_directory(DicomData_dir, output_dir,True,True,subject)
        print(DicomData_dir + " Done")

