import numpy as np
import scipy as sc
from skimage import io, img_as_ubyte
import os
import fbpca



#Search for all files
FileNames = os.listdir('./RawData')
NumberofFiles = len(FileNames)
print(NumberofFiles)

ImageSize = np.shape(io.imread("./RawData/"+FileNames[1],as_gray=True))

ImageVectors = [] #np.zeros([2032,20480])#[(ImageSize[1]*ImageSize[2]),(ImageSize[2]*NumberofFiles)])

i = 0
while i < NumberofFiles:
	Image = io.imread("./RawData/"+FileNames[i],as_gray=True,) # Read the .TIF image as grayscale (required to keep the same data size)
	Imagebyte = img_as_ubyte(Image)
	ImageVector = np.reshape(Imagebyte,[1,(ImageSize[0]*ImageSize[1])])
	ImageVectors.extend(ImageVector)
	i += 1


fbpca.svd(ImageVectors)

print(np.size(ImageVectors))
