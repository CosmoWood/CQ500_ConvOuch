import nibabel as nib
from dipy.align.reslice import reslice
import os
from matplotlib import pyplot as plt

nifti_folder = './Nifti_Files'

#here we should provide a list of scans that need to be resliced
#or we just list everything in the folder
scan_list = os.listdir(nifti_folder)

#the resolution of reslicing
new_zooms = (1., 1., 5.)

for i in scan_list:
    if i.startswith('CQ500'):
        #scan_name = (nifti_folder + '/' + i + '/' + i + '.nii.gz')
        scan_name = (nifti_folder + '/' + i)
        print(scan_name)
        img = nib.load(scan_name)
        data = img.get_fdata()

        #maybe we reslcie everything?

        affine = img.affine
        zooms = img.header.get_zooms()[:3]

        data2, affine2 = reslice(data, affine, zooms, new_zooms)

        img2 = nib.Nifti1Image(data2, affine2)

        new_name = scan_name[:-7] + '_5mm' + '.nii'
        nib.save(img2, new_name)