import numpy as np
import scipy as sc
from scipy.sparse.linalg import svds #because these images are hyyuuuuuuuuuuuuuugeeeee and total svd methods cannot handle this amount of data
from scipy.sparse import csr_matrix
from scipy import misc
import os
import fnmatch
import time
import cv2

#Put this python script in the run folder, and create a folder in the run folder called "Processed," it will read from "RawData" and write into "Processed"


#Input Variables
NumOfSubtractionModes = 1 #more modes means more time computing the SVD as well as doing background subtraction for each image
NumOfImagesPerSVDCycle = 100 #must be an exact divisor of the total number of images in each frame(i.e. the run was 1500 captures, use 100, 150, 300, 500, etc.)
Frames = ['*V3VLA*','*V3VLB*','*V3VRA*','*V3VRB*','*V3VTA*','*V3VTB*'] #Frame name keyphrases that it searches for


os.system('cls' if os.name == 'nt' else 'clear')
NumOfFrames = len(Frames)
FileNames = sorted(os.listdir('./RawData'))
NumOfCycles = (len(FileNames)/(NumOfImagesPerSVDCycle*NumOfFrames))
for CycleNumber in range(1,(np.floor(NumOfCycles)).astype(int) + 1): # Cycle through all the files in NumOfImagesPerSVDCycle increments, np.floor is for odd numbers of files that don't match multiples of NumImagesPerSVDCycle
	CycleNames = FileNames[(((NumOfImagesPerSVDCycle*NumOfFrames)*CycleNumber)-(NumOfFrames*NumOfImagesPerSVDCycle)):((NumOfImagesPerSVDCycle*NumOfFrames)*CycleNumber)] #all the names for one cycle, read as 1LA,1LB,1RA,1RB,1TA,1TB,2LA,etc.
	for frame in range(NumOfFrames): # Cycle through all the frames in a given cycle
		FrameNames = fnmatch.filter(CycleNames,Frames[frame]) #find all the names for a given frame in the cycle
		NumberofFiles = len(FrameNames)
		ImageSize = np.shape(cv2.imread("./RawData/" + FrameNames[0])) #test the first image for size
		ImageSize =ImageSize[0:2]
		ImageVectors = np.empty([NumberofFiles,ImageSize[0] * ImageSize[1]],dtype=np.uint8) #initialization before loop
		print("\n--------------------------Reading Images---------------------------\n")
		for ImageNumber in range(NumberofFiles):
			print("Reading:",FrameNames[ImageNumber])
			image = cv2.imread("./RawData/"+FrameNames[ImageNumber]).astype(np.uint8)
			ImageVectors[ImageNumber,:] = (np.ravel(image[:,:,0],"F")) #ravel/vectorize each 2D image to 1D and insert into the 2D matrix "ImageVectors"
		print("\n-----------------Performing Sparse SVD Algorithm-------------------\n")
		u, s, v = svds(csr_matrix.asfptype(np.transpose(ImageVectors)),k=NumOfSubtractionModes,which='LM')
		um,sm,vm = np.asmatrix(u),np.asmatrix(s),np.asmatrix(v) #move to matrix data type for processing
		del u,s,v #memory management
		SingleImageModes = np.empty([NumOfSubtractionModes,ImageSize[0],ImageSize[1]]) #initialization before loop
		print("\n--------------------------Writing Images---------------------------\n")
		for image in range(NumOfImagesPerSVDCycle):
			for modes in range(NumOfSubtractionModes):
				SingleImageModes[modes,:,:] = np.reshape(um[:,modes] * sm[:,modes] * vm[modes,image],[ImageSize[0],ImageSize[1]],order="F") #reshape the modes back into a stack of 2D background images
			FilteredImage = np.reshape(np.transpose(ImageVectors[image,:]),[ImageSize[0],ImageSize[1]],order="F") - np.sum(SingleImageModes,axis=0) #subtract the sum of all background modes from the original image
			FilteredImage[FilteredImage < 0] = 0 #clean up/delete all negative values, as those will wrap around when saving it in unsigned byte format
			STDFilter = 10*np.std(FilteredImage[FilteredImage != 0]) #10 is chosen arbitrarily for the data, it seems to work well for PIV data
			FilteredImage[FilteredImage > STDFilter] = STDFilter #remove all values that are greater than the standard deviation based filter (this helps to eliminate glare points in the image)
			print("Writing:",FrameNames[image])
			cv2.imwrite('./Processed/PCA'+FrameNames[image],misc.bytescale(FilteredImage)) #save the image in byte format
		timeleft = (time.clock() * ((np.floor(NumOfCycles)*NumOfFrames) / ( ((CycleNumber-1)*NumOfFrames)+(frame+1) )))-time.clock()
		os.system('cls' if os.name == 'nt' else 'clear')
		print("\nTime to completion:",format(timeleft/60, '.4f'),"minutes")
print('\n\n\nFinished! :)')
