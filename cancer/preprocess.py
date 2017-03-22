# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:22:00 2017

@author: hubj
"""

# %matplotlib inline
import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
# import matplotlib.pyplot as plt
import pickle

# from skimage import measure, morphology
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, new_shape):
    current_shape = np.array(image.shape)
    resize_factor = new_shape / current_shape
    return scipy.ndimage.interpolation.zoom(image, resize_factor)

def plot_3d(image, threshold=-300, alpha=0.1):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

INPUT_FOLDER = 'data/'

def crop_rip_cage(hu, threshold):
	x0_mid = int(hu.shape[0]/2)
	x1_min = None
	x1_max = None
	for x1 in range(hu.shape[1]):
		# print(x0_mid)
		# print(x1)
		# print(hu[x0_mid:, x1, :].shape)
		# print(hu[x0_mid:, x1, :])
		# print(hu[x0_mid:, x1, :]>threshold)
		if (hu[x0_mid:,x1,:]>threshold).any():
			x1_min = x1
			break
	for x1 in range(hu.shape[1]-1,0,-1):
		if (hu[x0_mid:,x1,:]>threshold).any():
			x1_max = x1
			break
	assert(x1_min<x1_max)
	x2_min = None
	x2_max = None
	for x2 in range(hu.shape[2]):
		if (hu[x0_mid:,:,x2]>threshold).any():
			x2_min = x2
			break
	for x2 in range(hu.shape[2]-1,0,-1):
		if (hu[x0_mid:,:,x2]>threshold).any():
			x2_max = x2
			break
	assert(x2_min<x2_max)

	x0_min = None
	x0_max = None
	for x0 in range(hu.shape[0]):
		if (hu[x0,:,:]>threshold).any():
			x0_min = x0
			break
	for x0 in range(hu.shape[0]-1,0,-1):
		if (hu[x0,:,:]>threshold).any():
			x0_max = x0
			break
	assert(x0_min<x0_max)

	# print('x0', x0_min, x0_max)
	# print('x1', x1_min, x1_max)
	# print('x2', x2_min, x2_max)

#	x0_mid = (x0_max+x0_min)/2
#	x1_mid = (x1_max+x1_min)/2
#	x2_mid = (x2_max+x2_min)/2
#	plt.figure()
#	plt.imshow(hu[x0_min + (x0_max-x0_min)/3, x1_min:x1_max, x2_min:x2_max])
##	plt.figure()
##	plt.imshow(hu[x0_mid, x1_min:x1_max, x2_min:x2_max])
#	plt.figure()
#	plt.imshow(hu[x0_min + (2*(x0_max-x0_min))/3, x1_min:x1_max, x2_min:x2_max])
#	plt.figure()
#	plt.imshow(hu[x0_min:x0_max, x1_mid, x2_min:x2_max])
#	plt.figure()
#	plt.imshow(hu[x0_min:x0_max, x1_min:x1_max, x2_min + (x2_max-x2_min)/3])
##	plt.figure()
##	plt.imshow(hu[x0_min:x0_max, x1_min:x1_max, x2_mid])
#	plt.figure()
#	plt.imshow(hu[x0_min:x0_max, x1_min:x1_max, x2_min + ((x2_max - x2_min)*2)/3])

	# plt.imshow(resample(hu, [123, 123, 123]))
	return x0_max-x0_min, x1_max-x1_min,x2_max-x2_min


def preprocess(patId):
	first_patient = load_scan(INPUT_FOLDER + patId)
	first_patient_pixels = get_pixels_hu(first_patient)

	cropped_dimensions = crop_rip_cage(first_patient_pixels, 600)
	pix_resampled = resample(first_patient_pixels, [93, 218, 356])
	return pix_resampled
	# return cropped_dimensions

# Some constants
patients = os.listdir(INPUT_FOLDER)
# patients = [patients[0]]
patients.sort()
data = None
# min_x0 = min_x1 = min_x2 = 987656432
# max_x0 = max_x1 = max_x2 = 0

for i, patId in enumerate(patients):
	print("processing patient {0:s} {1:d}/{2:d}".format(patId, i+1, len(patients)))
	data = preprocess(patId)
	with open("preprocessed/{0:s}.pickle".format(patId), "wb") as f:
		pickle.dump(data, f)

	# min_x0 = min(data[0], min_x0)
	# min_x1 = min(data[1], min_x1)
	# min_x2 = min(data[2], min_x2)
	# max_x0 = max(data[0], max_x0)
	# max_x1 = max(data[1], max_x1)
	# max_x2 = max(data[2], max_x2)

# print('min_x0', min_x0)
# print('min_x1', min_x1)
# print('min_x2', min_x2)
# print('max_x0', max_x0)
# print('max_x1', max_x1)
# print('max_x2', max_x2)


#	with open("{0:s}.pickle".format(patId), "w") as f:
#		pickle.dump(data, f)
#sick_data = preprocess('0acbebb8d463b4b9ca88cf38431aac69')
#healthy1_data = main(patients[1])

