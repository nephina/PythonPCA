import numpy as np
from numpy import ndarray
import scipy as sc
from scipy.sparse.linalg import svds, eigs
from scipy.sparse import csr_matrix
from scipy import misc
from skimage import io, img_as_ubyte
import skimage
import os
import fnmatch



# Input Variables

NumOfSubtractionModes = 1 # This doesn't work for anything but 1 mode yet
NumOfImagesPerSVDCycle = 100


Frames = ['*V3VLA*','*V3VLB*','*V3VRA*','*V3VRB*','*V3VTA*','*V3VTB*']


#Search for all files
FileNames = sorted(os.listdir('./RawData'))
NumOfCycles = (len(FileNames)/(NumOfImagesPerSVDCycle*6))

CycleNumber = 1
while CycleNumber <= NumOfCycles: # Cycle through all the files in NumOfImagesPerSVDCycle increments
	CycleNames = FileNames[(((NumOfImagesPerSVDCycle*6)*CycleNumber)-(6*NumOfImagesPerSVDCycle)):((NumOfImagesPerSVDCycle*6)*CycleNumber)] #all the names for one cycle, read in 1LA,1LB,1RA,1RB,1TA,1TB,2LA,etc.

	frame = 0
	while frame < 6: # Cycle through all the frames in a given cycle
		FrameNames = fnmatch.filter(CycleNames,Frames[frame]) #all the names for a given frame in the 
		NumberofFiles = len(FrameNames)

		ImageSize = np.shape(io.imread("./RawData/"+FrameNames[0],as_gray=True)) #test the first image for size
		FirstImage = io.imread("./RawData/"+FrameNames[0]) #read the first image
		ImageVectors = np.ravel(FirstImage[:,:,0],"F") #setup the first image in "imagevectors" in order to make the np.vstack work properly

		ImageNumber = 1 #starts out at 1 because we already got the first one to start off our ImageVectors matrix
		while ImageNumber < NumberofFiles: # Cycle through all the images in a given frame of a given cycle
			Image = io.imread("./RawData/"+FrameNames[ImageNumber])
			ImageVectors = np.vstack((ImageVectors,np.ravel(Image[:,:,0],"F")))

			ImageNumber += 1

		ImageVectors = np.transpose(ImageVectors)
		Images = np.reshape(ImageVectors,[ImageSize[0],ImageSize[1],NumberofFiles],order="F")

		u, s, v = sc.sparse.linalg.svds(sc.sparse.csr_matrix.asfptype(ImageVectors),k=NumOfSubtractionModes,which='LM')
		um = np.asmatrix(u)
		sm = np.asmatrix(s)
		vm = np.asmatrix(v)


		Mode1 = [um[:,0]*sm[:,0]*vm[0,:]]
		Processed = Images - np.reshape(Mode1,[ImageSize[0],ImageSize[1],NumberofFiles],order='F')

		OutputNum = 0
		while OutputNum < NumOfImagesPerSVDCycle:
			print(FrameNames[OutputNum])
			tempbyteimage = Processed[:,:,OutputNum] - np.median(Processed[:,:,OutputNum]) #subtract the median to get a dark background
			tempbyteimage[tempbyteimage < 0] = 0 #remove anything below 0, this way the median subtraction results that are less than 0 are discarded
			misc.imsave('./Processed/PCA'+FrameNames[OutputNum],misc.bytescale(tempbyteimage)) #write the output image to the processed folder and add "PCA," scale the floats to bytes for 0-255 grayscale

			OutputNum += 1
		
		frame += 1
	
	CycleNumber +=1

print('\n\n\nFinished! :)')





