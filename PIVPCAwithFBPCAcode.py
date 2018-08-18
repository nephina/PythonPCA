import numpy as np
import scipy as sc
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy import misc
from skimage import io
import os
import fnmatch


NumOfSubtractionModes = 1
NumOfImagesPerSVDCycle = 250


def uint8load(f):
	return (255*io.imread(f,as_gray=True)).astype(np.uint8)
Frames = ['*V3VLA*','*V3VLB*','*V3VRA*','*V3VRB*','*V3VTA*','*V3VTB*']
FileNames = sorted(os.listdir('./RawData'))
NumOfCycles = (len(FileNames)/(NumOfImagesPerSVDCycle*6))
CycleNumber = 1
while CycleNumber <= NumOfCycles: # Cycle through all the files in NumOfImagesPerSVDCycle increments
	CycleNames = FileNames[(((NumOfImagesPerSVDCycle*6)*CycleNumber)-(6*NumOfImagesPerSVDCycle)):((NumOfImagesPerSVDCycle*6)*CycleNumber)] #all the names for one cycle, read as 1LA,1LB,1RA,1RB,1TA,1TB,2LA,etc.
	for frame in range(6): # Cycle through all the frames in a given cycle
		FrameNames = fnmatch.filter(CycleNames,Frames[frame]) #find all the names for a given frame in the cycle
		NumberofFiles = len(FrameNames)
		ImageSize = np.shape(io.imread("./RawData/"+FrameNames[0],as_gray=True)) #test the first image for size
		ImageVectors = np.empty([NumberofFiles,ImageSize[0]*ImageSize[1]],dtype=np.uint8) #loop initialization
		print("\n--------------------------Reading Images---------------------------\n")
		ImageCollection = io.concatenate_images(io.ImageCollection(["./RawData/"+x for x in FrameNames],load_func=uint8load)) #read all images in a given cycle and frame at one time, reading them in byte format
		for ImageNumber in range(NumberofFiles):
			ImageVectors[ImageNumber,:] = (np.ravel(ImageCollection[ImageNumber,:,:],"F")).astype(np.uint8) #ravel/vectorize each 2D image to 1D and insert into the 2D stack "ImageVectors"
			print("Vectorizing Image: ",FrameNames[ImageNumber])
		del ImageCollection #memory management
		print("\n-----------------Performing Sparse SVD Algorithm-------------------\n")
		u, s, v = sc.sparse.linalg.svds(csr_matrix.asfptype(np.transpose(ImageVectors)),k=NumOfSubtractionModes,which='LM')
		um,sm,vm = np.asmatrix(u),np.asmatrix(s),np.asmatrix(v) #move to matrix form for processing
		del u,s,v #memory management
		SingleImageModes = np.empty([NumOfSubtractionModes,ImageSize[0],ImageSize[1]]) #loop initialization
		for image in range(NumOfImagesPerSVDCycle):
			for modes in range(NumOfSubtractionModes):
				SingleImageModes[modes,:,:] = np.reshape(um[:,modes]*sm[:,modes]*vm[modes,image],[ImageSize[0],ImageSize[1]],order="F") #reshape the modes to resemble a stack of 2D background images
			FilteredImage = np.reshape(np.transpose(ImageVectors[image,:]),[ImageSize[0],ImageSize[1]],order="F") - np.sum(SingleImageModes,axis=0) #subtract the sum of all background modes from the original image
			FilteredImage = FilteredImage-np.mean(FilteredImage) #subtract the mean to give a dark background
			FilteredImage[FilteredImage < 0] = 0 #clean up/delete all negative values, as those will mess with saving it in byte format
			print("Writing: PCA",FrameNames[image])
			misc.imsave('./Processed/PCA'+FrameNames[image],(FilteredImage).astype(np.uint8)) #save the image in byte format
	CycleNumber +=1
print('\n\n\nFinished! :)')