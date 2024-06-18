#!/usr/bin/env python
# %% Import libraries
from __future__ import print_function

import logging
import os

import six
from scipy.io import loadmat
import radiomics
from radiomics import featureextractor, getFeatureClasses, shape2D
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from radiomics import shape2D
import pandas as pd


# %% Make a 2D analysis -> Healthy patient

path_healthy = '/home/hoyeh3/GitHub-wsl2/pyradiomics/XH_Texture_Analysis/Healthy_test'

# Import image
imagefile = 'VentilationImage.nii.gz'
image_path = os.path.join(path_healthy, imagefile)
imageName_healthy = nib.load(image_path).get_fdata()
imageName_healthy = np.rot90(imageName_healthy,3)

# Import mask
maskfile = 'MaskImage.nii.gz'
mask_path = os.path.join(path_healthy, maskfile)
maskName_healthy = nib.load(mask_path).get_fdata()
maskName_healthy = np.rot90(maskName_healthy,3)

# Import the ventilation defects from Matlab
defect_path = os.path.join(path_healthy, 'VDPThresholdAnalysis.mat')
threshold_mat = loadmat(defect_path)
defect = np.fliplr(np.array(threshold_mat['defectArray']))
defect[defect != 0] = 1
mask_defect_healthy = abs(maskName_healthy-defect)

# Create a folder to store all the *.nii files
xh_folder_healthy = os.path.join(path_healthy, r'2D_analysis')
try:
    os.mkdir(xh_folder_healthy)
    print("Folder %s created!" % xh_folder_healthy)
except FileExistsError:
    print("Folder %s already exists" % xh_folder_healthy)


# Separate the slices and the mask into independent *.nii files
for i in range(imageName_healthy.shape[2]):
    output = np.rot90(imageName_healthy[:,:,i],3)
    ni_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(ni_img, xh_folder_healthy+'/'+'VentilationImage_{0}.nii.gz'.format(i))

    output = np.rot90(maskName_healthy[:,:,i],3)
    ni_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(ni_img, xh_folder_healthy+'/'+'MaskImage_{0}.nii.gz'.format(i))

    output = np.rot90(mask_defect_healthy[:,:,i],3)
    ni_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(ni_img, xh_folder_healthy+'/'+'VentilationDefect_{0}.nii.gz'.format(i))

    print('VentilationImage_{0}.nii.gz, MaskImage_{0}.nii.gz & VentilationDefect_{0}.nii.gz'.format(i))
print(' ')

# # Sanity Check
# for i in range(imageName_healthy.shape[2]):
#     plt.figure()
#     plt.subplot(1,3,1)
#     plt.imshow(imageName_healthy[:,:,i])
#     plt.title(f"Image #{i}")
#     plt.subplot(1,3,2)
#     plt.imshow(defect[:,:,i])
#     plt.title(f"Defect #{i}")
#     plt.subplot(1,3,3)
#     plt.imshow(imageName_healthy[:,:,i]*mask_defect_healthy[:,:,i])
#     plt.title(f"Image, no defects #{i}")

# %% To obtain the features slice by slice, 2D.
flag = 0
for i in range(imageName_healthy.shape[2]):
    path_image = os.path.join(xh_folder_healthy,'VentilationImage_{0}.nii.gz').format(i) 
    path_defect = os.path.join(xh_folder_healthy,'VentilationDefect_{0}.nii.gz').format(i)

    imageName = sitk.ReadImage(path_image)
    defectName = sitk.ReadImage(path_defect)

    # Extract the features
    extractor = shape2D.RadiomicsShape2D(imageName,defectName)
    featureClasses = getFeatureClasses()

    print(f"Calculating features for ventilation slice: {i}")
    featureVector = extractor.execute()

    for featureName in featureVector.keys():
        print("Computed %s: %s" % (featureName, featureVector[featureName]))
        df = pd.DataFrame(featureVector.keys(),columns=['Features'])
        df_2 = pd.DataFrame(featureVector.values(),columns=['Measurements'])
    
    # Add all the measurements for each slice in different columns
    if flag == 0:
        df_merged_healthy = pd.concat([df, df_2], axis=1)
    else:
        df_merged_healthy = pd.concat([df_merged_healthy, df_2], axis=1)
    flag = 1
    
    print(' ')

    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(np.rot90(sitk.GetArrayFromImage(imageName),2), cmap="gray")
    plt.title(f"Ventilation #{i}")
    plt.subplot(1,3,2)
    plt.imshow(np.rot90(sitk.GetArrayFromImage(defectName),2), cmap="gray")        
    plt.title(f"Mask-defects #{i}")
    plt.subplot(1,3,3)
    plt.imshow(np.rot90(sitk.GetArrayFromImage(imageName*defectName),2), cmap="gray")        
    plt.title(f"Without defects #{i}")


# %% Make a 2D analysis -> Lam patient

path_Lam = '/home/hoyeh3/GitHub-wsl2/pyradiomics/XH_Texture_Analysis/Lam_test'

# Import image
imagefile = 'VentilationImage.nii.gz'
image_path = os.path.join(path_Lam, imagefile)
imageName_Lam = nib.load(image_path).get_fdata()
imageName_Lam = np.rot90(imageName_Lam,3)

# Import mask
maskfile = 'MaskImage.nii.gz'
mask_path = os.path.join(path_Lam, maskfile)
maskName_Lam = nib.load(mask_path).get_fdata()
maskName_Lam = np.rot90(maskName_Lam,3)

# Import the ventilation defects from Matlab
defect_path = os.path.join(path_Lam, 'VDPThresholdAnalysis.mat')
threshold_mat = loadmat(defect_path)
defect = np.fliplr(np.array(threshold_mat['defectArray']))
defect[defect != 0] = 1
mask_defect_Lam = abs(maskName_Lam-defect)

# Create a folder to store all the *.nii files
xh_folder_Lam = os.path.join(path_Lam, r'2D_analysis')
try:
    os.mkdir(xh_folder_Lam)
    print("Folder %s created!" % xh_folder_Lam)
except FileExistsError:
    print("Folder %s already exists" % xh_folder_Lam)


# Separate the slices and the mask into independent *.nii files
for i in range(imageName_Lam.shape[2]):
    output = np.rot90(imageName_Lam[:,:,i],3)
    ni_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(ni_img, xh_folder_Lam+'/'+'VentilationImage_{0}.nii.gz'.format(i))

    output = np.rot90(maskName_Lam[:,:,i],3)
    ni_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(ni_img, xh_folder_Lam+'/'+'MaskImage_{0}.nii.gz'.format(i))

    output = np.rot90(mask_defect_Lam[:,:,i],3)
    ni_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(ni_img, xh_folder_Lam+'/'+'VentilationDefect_{0}.nii.gz'.format(i))

    print('VentilationImage_{0}.nii.gz, MaskImage_{0}.nii.gz & VentilationDefect_{0}.nii.gz'.format(i))
print(' ')

# # Sanity Check
# for i in range(imageName_Lam.shape[2]):
#     plt.figure()
#     plt.subplot(1,3,1)
#     plt.imshow(imageName_Lam[:,:,i])
#     plt.title(f"Image #{i}")
#     plt.subplot(1,3,2)
#     plt.imshow(defect[:,:,i])
#     plt.title(f"Defect #{i}")
#     plt.subplot(1,3,3)
#     plt.imshow(imageName_Lam[:,:,i]*mask_defect_control[:,:,i])
#     plt.title(f"Image, no defects #{i}")

# %% To obtain the features slice by slice, 2D.
flag = 0
for i in range(imageName_Lam.shape[2]):
    path_image = os.path.join(xh_folder_Lam,'VentilationImage_{0}.nii.gz').format(i) 
    path_defect = os.path.join(xh_folder_Lam,'VentilationDefect_{0}.nii.gz').format(i)

    imageName = sitk.ReadImage(path_image)
    defectName = sitk.ReadImage(path_defect)

    # Extract the features
    extractor = shape2D.RadiomicsShape2D(imageName,defectName)
    featureClasses = getFeatureClasses()

    print(f"Calculating features for ventilation slice: {i}")
    featureVector = extractor.execute()

    for featureName in featureVector.keys():
        print("Computed %s: %s" % (featureName, featureVector[featureName]))
        df = pd.DataFrame(featureVector.keys(),columns=['Features'])
        df_2 = pd.DataFrame(featureVector.values(),columns=['Measurements'])
    
    # Add all the measurements for each slice in different columns
    if flag == 0:
        df_merged_Lam = pd.concat([df, df_2], axis=1)
    else:
        df_merged_Lam = pd.concat([df_merged_Lam, df_2], axis=1)
    flag = 1

    print(' ')

    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(np.rot90(sitk.GetArrayFromImage(imageName),2), cmap="gray")
    plt.title(f"Ventilation #{i}")
    plt.subplot(1,3,2)
    plt.imshow(np.rot90(sitk.GetArrayFromImage(defectName),2), cmap="gray")        
    plt.title(f"Mask-defects #{i}")
    plt.subplot(1,3,3)
    plt.imshow(np.rot90(sitk.GetArrayFromImage(imageName*defectName),2), cmap="gray")        
    plt.title(f"Without defects #{i}")



# %% Make a 2D analysis -> Lam patchy patient

path_patchy = '/home/hoyeh3/GitHub-wsl2/pyradiomics/XH_Texture_Analysis/Lam_test_patchy'

# Import image
imagefile = 'VentilationImage.nii.gz'
image_path = os.path.join(path_patchy, imagefile)
imageName_patchy = nib.load(image_path).get_fdata()
imageName_patchy = np.rot90(imageName_patchy,3)

# Import mask
maskfile = 'MaskImage.nii.gz'
mask_path = os.path.join(path_patchy, maskfile)
maskName_patchy = nib.load(mask_path).get_fdata()
maskName_patchy = np.rot90(maskName_patchy,3)

# Import the ventilation defects from Matlab
defect_path = os.path.join(path_patchy, 'VDPThresholdAnalysis.mat')
threshold_mat = loadmat(defect_path)
defect = np.fliplr(np.array(threshold_mat['defectArray']))
defect[defect != 0] = 1
mask_defect_patchy = abs(maskName_patchy-defect)

# Create a folder to store all the *.nii files
xh_folder_patchy = os.path.join(path_patchy, r'2D_analysis')
try:
    os.mkdir(xh_folder_patchy)
    print("Folder %s created!" % xh_folder_patchy)
except FileExistsError:
    print("Folder %s already exists" % xh_folder_patchy)


# Separate the slices and the mask into independent *.nii files
for i in range(imageName_patchy.shape[2]):
    output = np.rot90(imageName_patchy[:,:,i],3)
    ni_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(ni_img, xh_folder_patchy+'/'+'VentilationImage_{0}.nii.gz'.format(i))

    output = np.rot90(maskName_patchy[:,:,i],3)
    ni_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(ni_img, xh_folder_patchy+'/'+'MaskImage_{0}.nii.gz'.format(i))

    output = np.rot90(mask_defect_patchy[:,:,i],3)
    ni_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(ni_img, xh_folder_patchy+'/'+'VentilationDefect_{0}.nii.gz'.format(i))

    print('VentilationImage_{0}.nii.gz, MaskImage_{0}.nii.gz & VentilationDefect_{0}.nii.gz'.format(i))
print(' ')

# # Sanity Check
# for i in range(imageName_Lam.shape[2]):
#     plt.figure()
#     plt.subplot(1,3,1)
#     plt.imshow(imageName_Lam[:,:,i])
#     plt.title(f"Image #{i}")
#     plt.subplot(1,3,2)
#     plt.imshow(defect[:,:,i])
#     plt.title(f"Defect #{i}")
#     plt.subplot(1,3,3)
#     plt.imshow(imageName_Lam[:,:,i]*mask_defect_control[:,:,i])
#     plt.title(f"Image, no defects #{i}")

# %% To obtain the features slice by slice, 2D.

# Main path
path = '/home/hoyeh3/GitHub-wsl2/pyradiomics/XH_Texture_Analysis'

flag = 0
for i in range(imageName_patchy.shape[2]):
    path_image = os.path.join(xh_folder_patchy,'VentilationImage_{0}.nii.gz').format(i) 
    path_defect = os.path.join(xh_folder_patchy,'VentilationDefect_{0}.nii.gz').format(i)

    imageName = sitk.ReadImage(path_image)
    defectName = sitk.ReadImage(path_defect)

    # Extract the features
    extractor = shape2D.RadiomicsShape2D(imageName,defectName)
    featureClasses = getFeatureClasses()

    print(f"Calculating features for ventilation slice: {i}")
    featureVector = extractor.execute()

    for featureName in featureVector.keys():
        print("Computed %s: %s" % (featureName, featureVector[featureName]))
        df = pd.DataFrame(featureVector.keys(),columns=['Features'])
        df_2 = pd.DataFrame(featureVector.values(),columns=['Measurements'])

    # Add all the measurements for each slice in different columns
    if flag == 0:
        df_merged_patchy = pd.concat([df, df_2], axis=1)
    else:
        df_merged_patchy = pd.concat([df_merged_patchy, df_2], axis=1)
    flag = 1

    print(' ')

    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(np.rot90(sitk.GetArrayFromImage(imageName),2), cmap="gray")
    plt.title(f"Ventilation #{i}")
    plt.subplot(1,3,2)
    plt.imshow(np.rot90(sitk.GetArrayFromImage(defectName),2), cmap="gray")        
    plt.title(f"Mask-defects #{i}")
    plt.subplot(1,3,3)
    plt.imshow(np.rot90(sitk.GetArrayFromImage(imageName*defectName),2), cmap="gray")        
    plt.title(f"Without defects #{i}")

# %% Mean, std, median and store data in Excel document

# Remove any column with NaN values
df_merged_healthy = df_merged_healthy.dropna(axis=1, how='any')
df_merged_Lam = df_merged_Lam.dropna(axis=1, how='any')
df_merged_patchy = df_merged_patchy.dropna(axis=1, how='any')

# Obtain the mean 
df_healthy_mean = df_merged_healthy['Measurements'].mean(axis=1, skipna = True)
df_Lam_mean = df_merged_Lam['Measurements'].mean(axis=1, skipna = True)
df_patchy_mean = df_merged_patchy['Measurements'].mean(axis=1, skipna = True)

df_healthy_mean = pd.DataFrame(df_patchy_mean,columns=['Mean'])
df_Lam_mean = pd.DataFrame(df_Lam_mean,columns=['Mean'])
df_patchy_mean = pd.DataFrame(df_patchy_mean,columns=['Mean'])

# Obtain the std
df_healthy_std = df_merged_healthy['Measurements'].std(axis=1, skipna = True)
df_Lam_std = df_merged_Lam['Measurements'].std(axis=1, skipna = True)
df_patchy_std = df_merged_patchy['Measurements'].std(axis=1, skipna = True)

df_healthy_std = pd.DataFrame(df_healthy_std,columns=['SD'])
df_Lam_std = pd.DataFrame(df_Lam_std,columns=['SD'])
df_patchy_std = pd.DataFrame(df_patchy_std,columns=['SD'])

# Obtain median
df_healthy_median = np.median(df_merged_healthy['Measurements'].values,axis=1)
df_Lam_median = np.median(df_merged_Lam['Measurements'].values,axis=1)
df_patchy_median = np.median(df_merged_patchy['Measurements'].values,axis=1)

df_healthy_median = pd.DataFrame(df_healthy_median,columns=['Median'])
df_Lam_median = pd.DataFrame(df_Lam_median,columns=['Median'])
df_patchy_median = pd.DataFrame(df_patchy_median,columns=['Median'])

# Join the Mean SD, and median to the original feature dataframe
df_merged_healthy = pd.concat([df_merged_healthy, df_healthy_mean,df_healthy_std,df_healthy_median], axis=1)
df_merged_Lam = pd.concat([df_merged_Lam, df_Lam_mean,df_Lam_std,df_Lam_median], axis=1)
df_merged_patchy = pd.concat([df_merged_patchy, df_patchy_mean,df_patchy_std,df_patchy_median], axis=1)

# Store data in an Excel sheet
with pd.ExcelWriter(path+'/Features_2D.xlsx') as writer:
    df_merged_healthy.to_excel(writer, sheet_name="Healthy", index=False)
    df_merged_Lam.to_excel(writer, sheet_name="Lam", index=False)
    df_merged_patchy.to_excel(writer, sheet_name="Patchy", index=False)