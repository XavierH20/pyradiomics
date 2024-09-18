#!/usr/bin/env python
# %% Import libraries
from __future__ import print_function

import logging
import os

import six

from radiomics import featureextractor, getFeatureClasses

import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.io import loadmat
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# %% Settings

# Get the location of the settings file
path = '/home/hoyeh3/GitHub-wsl2/pyradiomics/XH_Texture_Analysis'

# Initialize feature extractor using the settings file 'Params.yaml'
paramsFile = os.path.abspath(os.path.join(path,'Settings', 'Params.yaml'))
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
featureClasses = getFeatureClasses()

# # Enable features settings. i.e. Force 2D
# # https://pyradiomics.readthedocs.io/en/latest/features.html
settings = {}
settings['force2D'] = True
settings['force2Ddimension'] = 1  # 0 for axial slices (default), 1 for coronal slices, 2 for sagittal.
extractor = featureextractor.RadiomicsFeatureExtractor(**settings) 

# # Optionally enable some image types or filters:
# extractor.enableImageTypeByName('Wavelet')
# extractor.enableImageTypeByName('LoG', customArgs={'sigma':[3.0]})
# extractor.enableImageTypeByName('Square')
# extractor.enableImageTypeByName('SquareRoot')
# extractor.enableImageTypeByName('Exponential')
# extractor.enableImageTypeByName('Logarithm')

# Obtain radiomics properties
print("Active features:")
for cls, features in six.iteritems(extractor.enabledFeatures):
  if features is None or len(features) == 0:
    features = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
  for f in features:
    print('*',cls,f)
    # print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)

print('Extraction parameters:\n\t', extractor.settings)
print('Enabled features:\n\t', extractor.enabledFeatures)  # Still the default parameters

print('Enabled input images:')
for imageType in extractor.enabledImagetypes.keys():
    print('\t' + imageType)

# %% Import all files from patients
print("Importing Nifti files (ventilation and mask) and 'defects' file.")

# Control patient
path_control = os.path.join(path, 'Healthy_test')
data_control = os.path.join(path_control, 'VDPThresholdAnalysis.mat')
data_control = loadmat(data_control)

# Lam patient
path_Lam = os.path.join(path, 'Lam_test')
data_Lam = os.path.join(path_Lam, 'VDPThresholdAnalysis.mat')
data_Lam = loadmat(data_Lam)

# Very patchy Lam patient 
path_Lam_patchy = os.path.join(path, 'Lam_test_patchy')
data_Lam_patchy = os.path.join(path_Lam_patchy, 'VDPThresholdAnalysis.mat')
data_Lam_patchy = loadmat(data_Lam_patchy)


# %% Control patient
print('Control patient')

# Import images, mask and ventilation defects mask from Matlab
print(' *Import images, mask and ventilation defects mask')
imageName_control = np.array(data_control['MR'])
maskName_control = np.array(data_control['maskarray'])
defect = np.array(data_control['defectArray'])
defect[defect != 0] = 1

if imageName_control is None or maskName_control is None or defect is None:
  print('Error getting testcase!')
  exit()

# Sanity check
print(' *Show images, mask and defects (sanity check)')
mask_defect_control = np.zeros_like(imageName_control)

for numImage in range(imageName_control.shape[2]):
  # Substract from the ventilation defects from the originil mask
  mask_defect_control[:,:,numImage] = abs(maskName_control[:,:,numImage] - defect[:,:,numImage])

  # Sanity check to observe the original mask without the ventilation defects
  plt.figure(figsize=(10,10))
  plt.subplot(2,3,1)
  plt.imshow(imageName_control[:,:,numImage], cmap="gray")
  plt.title(f"Ventilation #{numImage}")
  plt.subplot(2,3,2)
  plt.imshow(maskName_control[:,:,numImage], cmap="gray")
  plt.title(f"Mask #{numImage}")
  plt.subplot(2,3,3)
  plt.imshow(imageName_control[:,:,numImage] * maskName_control[:,:,numImage])        
  plt.title(f"Mask and ventilation #{numImage}")
  plt.subplot(2,3,4)
  plt.title(f"Defect #{numImage}")
  plt.imshow(defect[:,:,numImage], cmap="gray")
  plt.subplot(2,3,5)
  plt.imshow(mask_defect_control[:,:,numImage], cmap="gray")        
  plt.title(f"Mask - defects #{numImage}")
  plt.subplot(2,3,6)
  plt.imshow(imageName_control[:,:,numImage] * mask_defect_control[:,:,numImage])        
  plt.title(f"Mask-defect #{numImage}")
  plt.tight_layout()

# Transform the array from the mask minus the vent. defect into an sitk Image
image_sitk_control = sitk.GetImageFromArray(imageName_control)
mask_defect_sitk_control = sitk.GetImageFromArray(mask_defect_control)

# Obtain radiomics properties
print("Calculating features")
featureVector = extractor.execute(image_sitk_control, mask_defect_sitk_control)

for featureName in featureVector.keys():
  print("Computed feature: %s " % (featureName))

df = pd.DataFrame(featureVector.keys(),columns = ['Features'])
df_2 = pd.DataFrame(featureVector.values(),columns = ['Measurements'])
df_merged_control = pd.concat([df, df_2], axis = 1)

# %% Lam patient 
print('Lam patient')

# Import images, mask and ventilation defects mask from Matlab
print(' *Import images, mask and ventilation defects mask')
imageName_Lam = np.array(data_Lam['MR'])
maskName_Lam = np.array(data_Lam['maskarray'])
defect = np.array(data_Lam['defectArray'])
defect[defect != 0] = 1

if imageName_Lam is None or maskName_Lam is None or defect is None:
  print('Error getting testcase!')
  exit()

# Sanity check
print(' *Show images, mask and defects (sanity check)')
mask_defect_Lam = np.zeros_like(imageName_Lam)

for numImage in range(imageName_Lam.shape[2]):
  # Substract from the ventilation defects from the originil mask
  mask_defect_Lam[:,:,numImage] = abs(maskName_Lam[:,:,numImage] - defect[:,:,numImage])

  # Sanity check to observe the original mask without the ventilation defects
  plt.figure(figsize=(10,10))
  plt.subplot(2,3,1)
  plt.imshow(imageName_Lam[:,:,numImage], cmap="gray")
  plt.title(f"Ventilation #{numImage}")
  plt.subplot(2,3,2)
  plt.imshow(maskName_Lam[:,:,numImage], cmap="gray")
  plt.title(f"Mask #{numImage}")
  plt.subplot(2,3,3)
  plt.imshow(imageName_Lam[:,:,numImage] * maskName_Lam[:,:,numImage])        
  plt.title(f"Mask and ventilation #{numImage}")
  plt.subplot(2,3,4)
  plt.title(f"Defect #{numImage}")
  plt.imshow(defect[:,:,numImage], cmap="gray")
  plt.subplot(2,3,5)
  plt.imshow(mask_defect_Lam[:,:,numImage], cmap="gray")        
  plt.title(f"Mask - defects #{numImage}")
  plt.subplot(2,3,6)
  plt.imshow(imageName_Lam[:,:,numImage] * mask_defect_Lam[:,:,numImage])        
  plt.title(f"Mask-defect #{numImage}")
  plt.tight_layout()

# Transform the array from the mask minus the vent. defect into an sitk Image
image_sitk_Lam = sitk.GetImageFromArray(imageName_Lam)
mask_defect_sitk_Lam = sitk.GetImageFromArray(mask_defect_Lam)

# Obtain radiomics properties
print("Calculating features")
featureVector = extractor.execute(image_sitk_Lam, mask_defect_sitk_Lam)

# for featureName in featureVector.keys():
#   print("Computed feature: %s " % (featureName))

df = pd.DataFrame(featureVector.keys(),columns = ['Features'])
df_2 = pd.DataFrame(featureVector.values(),columns = ['Measurements'])
df_merged_Lam = pd.concat([df, df_2], axis = 1)

# %% Very patchy Lam patient 
print('Lam patient patchy')

# Import images, mask and ventilation defects mask from Matlab
print(' *Import images, mask and ventilation defects mask')
imageName_Lam_patchy = np.array(data_Lam_patchy ['MR'])
maskName_Lam_patchy  = np.array(data_Lam_patchy ['maskarray'])
defect = np.array(data_Lam_patchy ['defectArray'])
defect[defect != 0] = 1

if imageName_Lam_patchy is None or maskName_Lam_patchy is None or defect is None:
  print('Error getting testcase!')
  exit()

# Sanity check
print(' *Show images, mask and defects (sanity check)')
mask_defect_Lam_patchy = np.zeros_like(imageName_Lam_patchy)

for numImage in range(imageName_Lam_patchy.shape[2]):
  # Substract from the ventilation defects from the originil mask
  mask_defect_Lam_patchy[:,:,numImage] = abs(maskName_Lam_patchy[:,:,numImage] - defect[:,:,numImage])

  # Sanity check to observe the original mask without the ventilation defects
  plt.figure(figsize=(10,10))
  plt.subplot(2,3,1)
  plt.imshow(imageName_Lam_patchy[:,:,numImage], cmap="gray")
  plt.title(f"Ventilation #{numImage}")
  plt.subplot(2,3,2)
  plt.imshow(maskName_Lam_patchy[:,:,numImage], cmap="gray")
  plt.title(f"Mask #{numImage}")
  plt.subplot(2,3,3)
  plt.imshow(imageName_Lam_patchy[:,:,numImage] * maskName_Lam_patchy[:,:,numImage])        
  plt.title(f"Mask and ventilation #{numImage}")
  plt.subplot(2,3,4)
  plt.title(f"Defect #{numImage}")
  plt.imshow(defect[:,:,numImage], cmap="gray")
  plt.subplot(2,3,5)
  plt.imshow(mask_defect_Lam_patchy[:,:,numImage], cmap="gray")        
  plt.title(f"Mask - defects #{numImage}")
  plt.subplot(2,3,6)
  plt.imshow(imageName_Lam_patchy[:,:,numImage] * mask_defect_Lam_patchy[:,:,numImage])        
  plt.title(f"Mask-defect #{numImage}")
  plt.tight_layout()

# Transform the array from the mask minus the vent. defect into an sitk Image
image_sitk_Lam_patchy = sitk.GetImageFromArray(imageName_Lam_patchy)
mask_defect_sitk_Lam_patchy = sitk.GetImageFromArray(mask_defect_Lam_patchy)

# Obtain radiomics properties
print("Calculating features")
featureVector = extractor.execute(image_sitk_Lam_patchy, mask_defect_sitk_Lam_patchy)

# for featureName in featureVector.keys():
#   print("Computed feature: %s " % (featureName))

df = pd.DataFrame(featureVector.keys(),columns = ['Features'])
df_2 = pd.DataFrame(featureVector.values(),columns = ['Measurements'])
df_merged_Lam_patchy = pd.concat([df, df_2], axis = 1)


# %% Store features into an Excel document

# Remove packages versions, original image and mask data, NOT radiomics
df_merged_control_filt = df_merged_control.iloc[22:]
df_merged_Lam_filt = df_merged_Lam.iloc[22:]
df_merged_Lam_filt_patchy = df_merged_Lam_patchy.iloc[22:]

print('Storing filtered features into an Excel document')
# Save data in excel
with pd.ExcelWriter(path+'/Features.xlsx') as writer:
    df_merged_control.to_excel(writer, sheet_name = "Control", index = False)
    df_merged_Lam.to_excel(writer, sheet_name = "Lam", index = False)
    df_merged_Lam_patchy.to_excel(writer, sheet_name = "Patchy", index = False)

    df_merged_control_filt.to_excel(writer, sheet_name = "Control_filt", index = False)
    df_merged_Lam_filt.to_excel(writer, sheet_name = "Lam_filt", index = False)
    df_merged_Lam_filt_patchy.to_excel(writer, sheet_name = "Patchy_filt", index = False)
