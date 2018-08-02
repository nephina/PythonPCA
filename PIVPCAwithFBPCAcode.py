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



Frames = ['*V3VLA*','*V3VLB*','*V3VRA*','*V3VRB*','*V3VTA*','*V3VTB*']

NumOfModes = 1

#Search for all files
FileNames = sorted(os.listdir('./RawData'))
frame = 0
while frame < 6:
	FrameNames = fnmatch.filter(FileNames,Frames[frame])
	NumberofFiles = len(FrameNames)
	NumOfCycles = NumberofFiles/50
	n = 1
	while n <= NumOfCycles:
		nFileNames = FrameNames[((50*n)-50):((50*n))]
		#print(nFileNames)
		nNumberofFiles = 50
		
		ImageSize = np.shape(io.imread("./RawData/"+nFileNames[0],as_gray=True))


		FirstImage = io.imread("./RawData/"+nFileNames[0])
		FirstImageByte = FirstImage[:,:,0] #img_as_ubyte(FirstImage[:,:,0])
		#print(np.shape(FirstImageByte))
		FirstImageVector = np.ravel(FirstImageByte,'F')
		ImageVectors = [FirstImageVector] #np.zeros([2032,20480])#[(ImageSize[1]*ImageSize[2]),(ImageSize[2]*NumberofFiles)])

		i = 1
		while i < nNumberofFiles:
			Image = io.imread("./RawData/"+nFileNames[i]) # Read the .TIF image as grayscale (required to keep the same data size)
			Imagebyte = Image[:,:,0] #img_as_ubyte(Image[:,:,0])
			ImageVector = np.ravel(Imagebyte,'F')
			ImageVectors = np.vstack( (ImageVectors,ImageVector) )
			i += 1
			#print(ImageVectors.shape)

		ImageVectors = np.transpose(ImageVectors)
		Images = np.reshape(ImageVectors,[ImageSize[0],ImageSize[1],nNumberofFiles],order='F')
		#print(np.shape(ImageVectors))
		#fbpca.svd(ImageVectors)
		u, s, v = sc.sparse.linalg.svds(sc.sparse.csr_matrix.asfptype(ImageVectors),k=NumOfModes,which='LM')
		um = np.asmatrix(u)
		sm = np.asmatrix(s)
		vm = np.asmatrix(v)

		Mode1 = [um[:,0]*sm[:,0]*vm[0,:]]
		print(np.shape(Mode1))
		Processed = Images - np.reshape(Mode1,[ImageSize[0],ImageSize[1],nNumberofFiles],order='F')
		print(np.shape(Images))
		print(np.shape(Processed))
		print(np.mean(Images))
		print(np.mean(Processed))
		#Mode1 = np.reshape(Mode1,[ImageSize[0],ImageSize[1],nNumberofFiles],order='F')
		#if NumOfModes > 1:
		#	Mode2 = [um[:,1]*sm[:,1]*vm[1,:]]
		#	Mode2 = np.reshape(Mode2,[ImageSize[0],ImageSize[1],nNumberofFiles],order='F')
		#if NumOfModes > 2:
		#	Mode3 = [um[:,2]*sm[:,2]*vm[2,:]]
		#	Mode3 = np.reshape(Mode3,[ImageSize[0],ImageSize[1],nNumberofFiles],order='F')
		#if NumOfModes > 3:
		#	Mode4 = [um[:,3]*sm[:,3]*vm[3,:]]
		#	Mode4 = np.reshape(Mode4,[ImageSize[0],ImageSize[1],nNumberofFiles],order='F')
		#if NumOfModes > 4:
		#	Mode5 = [um[:,4]*sm[:,4]*vm[4,:]]
		#	Mode5 = np.reshape(Mode5,[ImageSize[0],ImageSize[1],nNumberofFiles],order='F')
		#if NumOfModes > 5:
		#	Mode6 = [um[:,5]*sm[:,5]*vm[5,:]]
		#	Mode6 = np.reshape(Mode6,[ImageSize[0],ImageSize[1],nNumberofFiles],order='F')

		i = 0
		while i < 50:
			print(nFileNames[i])
			#tempbyteimage = misc.bytescale((((((Images[:,:,i]-Mode1[:,:,i])))))) #-Mode2[:,:,i])-Mode3[:,:,i])-Mode4[:,:,i])-Mode5[:,:,i])-Mode6[:,:,i])
			tempbyteimage = Processed[:,:,i] - np.median(Processed[:,:,i])
			#tempbytemean = np.mean(tempbyteimage[tempbyteimage > 0])
			#tempbytestd = np.std(tempbyteimage[tempbyteimage > 0])
			tempbyteimage[tempbyteimage < 0] = 0
			#tempbyteimage[tempbyteimage > 255] = 0
			#tempbyteimage[tempbyteimage > tempbytemean+(3*tempbytestd)] = 255
			#tempbyteimage[tempbyteimage > 0.95*np.max(tempbyteimage)] = 0
			#tempbyteimage[0,0]= 0
			#tempbyteimage[0,1] = 255
			misc.imsave('/media/alexa/Alexa/MCWAlexa/July19th/PCAProcessed/PCA'+nFileNames[i],misc.bytescale(tempbyteimage))
			i += 1
		n += 1
	frame += 1
print('\n\n\nFinished! :)')