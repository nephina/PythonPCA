import scipy as sc
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy import misc
from skimage import io
import skimage
import os
import fnmatch



# Input Variables

NumOfSubtractionModes = 1 # This doesn't work for anything but 1 mode yet
NumOfImagesPerSVDCycle = 200


Frames = ['*V3VLA*','*V3VLB*','*V3VRA*','*V3VRB*','*V3VTA*','*V3VTB*']


#Search for all files
FileNames = sorted(os.listdir('./RawData'))
NumOfCycles = (len(FileNames)/(NumOfImagesPerSVDCycle*6))

CycleNumber = 1
while CycleNumber <= NumOfCycles: # Cycle through all the files in NumOfImagesPerSVDCycle increments
	CycleNames = FileNames[(((NumOfImagesPerSVDCycle*6)*CycleNumber)-(6*NumOfImagesPerSVDCycle)):((NumOfImagesPerSVDCycle*6)*CycleNumber)] #all the names for one cycle, read in 1LA,1LB,1RA,1RB,1TA,1TB,2LA,etc.
	for frame in range(6): # Cycle through all the frames in a given cycle
		print("\n--------------------------Matching Names---------------------------\n")
		FrameNames = fnmatch.filter(CycleNames,Frames[frame]) #find all the names for a given frame in the cycle
		NumberofFiles = len(FrameNames)

		ImageSize = np.shape(io.imread("./RawData/"+FrameNames[0],as_gray=True)) #test the first image for size
		ImageVectors = np.empty([NumberofFiles,ImageSize[0]*ImageSize[1]]) #initialize the matrix for SVD
		print("\n--------------------------Reading Images---------------------------\n")
		ImageCollection = io.concatenate_images(io.ImageCollection(["./RawData/"+x for x in FrameNames],as_gray=True)) #read all images in a given cycle and frame at one time
		print("\n------------------Vectorizing Individual Images--------------------\n")
		for ImageNumber in range(NumberofFiles):
			ImageVectors[ImageNumber,:] = np.ravel(ImageCollection[ImageNumber,:,:],"F") #ravel/vectorize each 2D image to 1D and insert into the 2D stack "ImageVectors"
			print("Vectorizing Image: ",FrameNames[ImageNumber])
		print("\n-----------------Deleting Image Collection Files-------------------\n")
		del ImageCollection
		print("\n-----------------Performing Sparse SVD Algorithm-------------------\n")
		u, s, v = sc.sparse.linalg.svds(csr_matrix.asfptype(np.transpose(ImageVectors)),k=NumOfSubtractionModes,which='LM')
		print("\n---------------Writing SVD Results To Matrix Format----------------\n")
		um,sm,vm = np.asmatrix(u),np.asmatrix(s),np.asmatrix(v)
		print("\n---------------------Deleting SVD Raw Results----------------------\n")
		del u,s,v
		print("\n-----------------------Calculating Mode 1--------------------------\n")
		Mode1 = [um[:,0]*sm[:,0]*vm[0,:]] #multiply SVD results by each other in order to produce the background images in the same vectorized format of "ImageVectors"
		Mode1ImageFormat = np.reshape(Mode1,[ImageSize[0],ImageSize[1],NumberofFiles],order='F') #reshape the SVD vectorized stack results into a 3D stack of 2D images
		print("\n-------------------Deleting SVD Matrix Results---------------------\n")
		del um,sm,vm,Mode1
		Images = np.reshape(np.transpose(ImageVectors),[ImageSize[0],ImageSize[1],NumberofFiles],order="F") #reshape the original image data from "ImageVectors" in the same way
		print("\n---------------Deleting Image Vectorization Files------------------\n")
		del ImageVectors
		print("\n-------------------Generating Processed Images---------------------\n")
		Processed = Images - Mode1ImageFormat #subtract the background images from SVD from the original images, generating a filtered imageset
		del Mode1ImageFormat,Images
		OutputNum = 0
		while OutputNum < NumOfImagesPerSVDCycle:
			print("Writing: PCA",FrameNames[OutputNum])
			tempbyteimage = Processed[:,:,OutputNum] - np.mean(Processed[:,:,OutputNum]) #subtract the median to get a dark background
			tempbyteimage[tempbyteimage < 0] = 0 #remove anything below 0, this way the median subtraction results that are less than 0 are discarded
			misc.imsave('./Processed/PCA'+FrameNames[OutputNum],misc.bytescale(tempbyteimage)) #write the output image to the processed folder and add "PCA," scale the floats to bytes for 0-255 grayscale

			OutputNum += 1
		print("\n-------------------Deleting Processed Images-----------------------\n")
		del Processed
	
	CycleNumber +=1

print('\n\n\nFinished! :)')
