# -*- coding: iso8859-1 -*- 

## 
#   Utilisation de mamba
from mamba import *
from mambaComposed import *
import numpy as np    
from cvnumpy import *
from Analyz import *


def getArrayFromImage(imIn):
    """
    Creates an 2D array containing the same data as in 'imIn'. Only
    works for greyscale and 32-bit images. Returns the array.
    """
    if imIn.getDepth()==8:
        dtype = np.uint8
    elif imIn.getDepth()==32:
        dtype = np.uint32
    else:
        import mambaCore
        raiseExceptionOnError(mambaCore.ERR_BAD_DEPTH)
        
    (w,h) = imIn.getSize()
    # First extracting the raw data out of image imIn
    data = imIn.extractRaw()
    # creating an array with this data
    # At this step this is a one-dimensional array
    array1D = np.fromstring(data, dtype=dtype)
    # Reshaping it to the dimension of the image
    array2D = array1D.reshape((h,w))
    return array2D
    
def fillImageWithArray(array, imOut):
    """
    Fills image 'imOut' with the content of two dimensional 'array'. Only
    works for greyscale and 32-bit images.
    """
    # Checking depth 
    if (imOut.getDepth()==8 and array.dtype != np.uint8) or \
       (imOut.getDepth()==32 and array.dtype != np.uint32) or \
       (imOut.getDepth()==1):
        import mambaCore
        raiseExceptionOnError(mambaCore.ERR_BAD_DEPTH)
    
    # image size
    (wi,hi) = imOut.getSize()
    # array size
    (ha,wa) = array.shape
    
    # Checking the sizes
    if wa!=wi or ha!=hi:
		print( (wa,ha) , (wi,hi) )
		import mambaCore
		raiseExceptionOnError(mambaCore.ERR_BAD_SIZE)

    # Extracting the data out of the array and filling the image with it
    data = array.tostring()
    imOut.loadRaw(data)
	
def fillImageWithIplimage(cvImage):	
	imOut=imageMb( cvImage.width, cvImage.height, np.int(cvImage.depth))
	Arr_Image=cv2array(cvImage)[:,:,0]
	new_shape=imOut.getSize()
	Arr=np.zeros((new_shape[1], new_shape[0]),dtype=np.uint8)
	Arr[0:Arr_Image.shape[0], 0:Arr_Image.shape[1]]=Arr_Image
	fillImageWithArray(Arr, imOut)
	return imOut	
	
def granulo_func(cv_im, border=None):
	initial_shape=(cv_im.height, cv_im.width)
	im1 = fillImageWithIplimage(cv_im)
	# Defining working images.
	imWrk0 = imageMb(im1)
	imWrk1 = imageMb(im1)
	imWrk2 = imageMb(im1, 32)
	imWrk3 = imageMb(im1)

	# The initial image is filtered by an alternate filter by reconstruction.
	buildOpen(im1, imWrk0)
	buildClose(imWrk0, imWrk0)
	# The ultimate opening residual operator is applied to the image. In order
	# to obtain results of better quality, operators with dodecagons are used.
	ultimateIsotropicOpening(imWrk0, imWrk1, imWrk2)	
	copyBytePlane(imWrk2, 0, imWrk3)
	
	# histogram
	histo=getHistogram(imWrk3)
	granul_f=getArrayFromImage(imWrk3)	
	
	# if border!=None:
		# granul_f[border]=0
	print("close holes...")	
	granul_f = pymorph.close_holes(granul_f)		
	histo = pymorph.histogram(granul_f)
	
	if border!=None:
		im1 = fillImageWithIplimage(array2cv(border))
		buildOpen(im1, imWrk0)
		buildClose(imWrk0, imWrk0)

		ultimateIsotropicOpening(imWrk0, imWrk1, imWrk2)	
		copyBytePlane(imWrk2, 0, imWrk3)

		granul_f_border=getArrayFromImage(imWrk3)
		print("close holes...")	
		granul_f_border = pymorph.close_holes(granul_f_border)		
		histo_border = histo * 0
		histo_border[0:np.max(granul_f_border)+1] += pymorph.histogram(granul_f_border)	
		histo_border[0] = 0
	# histo_border=0
	
	# Display
	fig = pl.figure()
	fig.subplots_adjust(left=0.2, wspace=0.6)
	h=histo

	s = ME(histo)
	tmp=(histo-histo_border)*0.0
	tmp[0:s]=(histo-histo_border)[0:s]
	tmp[tmp<0]=0
	# histo=np.convolve( tmp, [1,1,1,1,1], 'same')/5

	ax1 = fig.add_subplot(211)
	ax1.plot(s*np.ones(np.max(histo)), np.arange(np.max(histo)),'g--')
	ax1.plot(histo_border,'y',label='hist fn granulometrique des bords')
	ax1.plot(histo,'r',label='hist fn granulometrique')
	ax1.set_title('histogramme de fn granulo')
	ax1.set_ylabel('Nombre pixels')
	ax1.set_xlabel('intensite pixel ou diametre element structurant')	
	ax1.text(s, 10, 'seuille='+repr(s),
			horizontalalignment='left',
			verticalalignment='top',
			bbox=dict(facecolor='yellow', alpha=0.3))	
	histo = tmp
	
	histo_cumul = np.cumsum(histo)
	ax3 = fig.add_subplot(212)
	ax3.plot(np.sum(histo)*0.5*np.ones(np.shape(histo)),'g--')
	ax3.plot(np.cumsum(h),'b', label='histo. cumulee fonction granulometrique')	
	ax3.plot(histo_cumul,'r', label='hist. cumulee fn. granulo. sans bord et seuillage valeurs perchées')
	ax3.set_title('histogramme cumule en taille')
	ax3.set_ylabel('Nombre pixels cumulee')
	ax3.set_xlabel('intensite pixel ou diametre element structurant')	
	
	# interpolation
	x=np.arange(len(histo_cumul))
	valeur= np.sum(histo)*0.5
	index = pl.find(np.convolve(histo_cumul<=valeur, [-1,1], 'same')==1)
	
	ax3.text(index, valeur, 'D50='+repr((int(index),int(index+1)))+' pixels',
			horizontalalignment='left',
			verticalalignment='bottom',
			bbox=dict(facecolor='yellow', alpha=0.3))	
	fig.show()	
	print("D50=%i"%(index))
	
	fond = granul_f > s
	cv_fond = array2cv(fond*255)
	cv.SaveImage("fond.jpg", cv_fond)
	
	return granul_f, histo, fig 
			
	
	
def Segmentation(cv_im, cv_edge=None):
	from mamba import *
	from mambaComposed import *
	initial_shape=(cv_im.height, cv_im.width)
	im1 = fillImageWithIplimage(cv_im)
	# Defining working images.
	imWrk0 = imageMb(im1)
	imWrk1 = imageMb(im1)
	imWrk2 = imageMb(im1, 32)
	imWrk3 = imageMb(im1)
	imWrk4 = imageMb(im1)
	imWrk5 = imageMb(im1, 1)
	imWrk6 = imageMb(im1)
	imWrk7 = imageMb(im1)
	imWrk8 = imageMb(im1, 1)
	imWrk9 = imageMb(im1, 1)
		
	
	# The initial image is filtered by an alternate filter by reconstruction.
	buildOpen(im1, imWrk0)
	buildClose(imWrk0, imWrk0)
	
	# The ultimate opening residual operator is applied to the image. In order
	# to obtain results of better quality, operators with dodecagons are used.
	ultimateIsotropicOpening(imWrk0, imWrk1, imWrk2)
	# imWrk3.save('current_images/grain.png')	
	copyBytePlane(imWrk2, 0, imWrk3)
	# This image is saved with a color palette.
	# imWrk3.setPalette(patchwork)
	# imWrk3.save('current_images/granu.png')
	# The flat zones of the granulometric image are extracted (these zones have
	# a gradient equal to zero).
	gradient(imWrk3, imWrk4)
	threshold(imWrk4, imWrk5, 1, 255)
	negate(imWrk5, imWrk5)
	# Holes in these zones are filled (they correspond to artifacts).
	closeHoles(imWrk5, imWrk5)
	# The real size of these flat zones is determined by a dodecagonal distance
	# function.
	isotropicDistance(imWrk5, imWrk2, edge=FILLED)
	copyBytePlane(imWrk2, 0, imWrk6)
	# We add one to correct the bias brought by the gradient.
	addConst(imWrk6, 1, imWrk6)
	# The real size is compared to the size given by the granulometric function.
	# When the real size is less than half the size of the granulometric function,
	# the corresponding flat zone cannot be considered as a marker of a block.
	divConst(imWrk3, 2, imWrk7)
	generateSupMask(imWrk6, imWrk7, imWrk8, False)
	# The extracted markers are filtered with an alternate filter by reconstruction
	# to connect the closest ones and to remove the smallest ones. 
	buildClose(imWrk8, imWrk9)
	buildOpen(imWrk8, imWrk9)
	# The result is saved.

	# The gradient of the original filtered image is computed.
	if cv_edge!=None:
		initial_shape=(cv_im.height, cv_im.width)
		im_edge = fillImageWithIplimage(cv_edge)
		imWrk1=imageMb(im_edge)
	else:
		gradient(imWrk0, imWrk1)
		# imWrk1.save('current_images/contour.png')
	
	# In order to get a better result, we must also introduce markers for the
	# background. These markers correspond the highest flat zones of the
	# granulometric function.
	# t = computeRange(imWrk3)[1]
	# threshold(imWrk3, imWrk5, t, 255)

	# Computing the automatic threshold image
	ME(imWrk3, imWrk5)
	# imWrk5.save('current_images/background.png')
	# A small correction is applied to insure that blocks markers and background
	# markers are not touching each other.
	dilate(imWrk5, imWrk8)
	diff(imWrk9, imWrk8, imWrk9)
	# To be sure that background markers mark only the background, they are reduced
	# to a point (after filling of their holes).
	closeHoles(imWrk5, imWrk5)
	thinD(imWrk5, imWrk5)
	# The rocks markers are labelled.
	nbStones = label(imWrk9, imWrk2)
	# We add 1 to the label values to let room for the background marker.
	add(imWrk2, imWrk9, imWrk2)
	# The background marker is added (label 1). Note that all the connected
	# components of the background marker share the same label value.
	add(imWrk2, imWrk5, imWrk2)
	# The watershed of the gradient is performed.
	# imWrk5.save('current_images/markers.png')	
	# markerControlledWatershed(imWrk1, imWrk2, imWrk1)
	markers = getArrayFromImage(imWrk2)[0:initial_shape[0],0:initial_shape[1]]

	basinSegment(imWrk1, imWrk2)
	copyBytePlane(imWrk2, 3, imWrk4)
	threshold(imWrk4, imWrk8, 0,0)
	# The background marker is used to build the catchment basins corresponding
	# to the background. Then, they are removed from the segmented image.
	build(imWrk8, imWrk5)
	diff(imWrk8, imWrk5, imWrk5)

	# The segmented image is saved.
	# imWrk8.save('current_images/segment.png')
	segmentation = getArrayFromImage(imWrk2)[0:initial_shape[0],0:initial_shape[1]]
	# Markers
	markers.dtype = np.float32
	print markers.dtype
	cv_markers = array2cv(markers)
	cv_markers_8bit = cv.CreateImage(cv.GetSize(cv_markers),  cv.IPL_DEPTH_8U, 1)
	cv.ConvertScale(cv_markers, cv_markers_8bit)
	
	
	segmentation = Enleve_bord(segmentation)
	
	segmentation = indice_circularite(segmentation, 0.1)	
	
	# segmentation.dtype = np.float32
	# cv_segmentation = array2cv(segmentation)
	
	# img = cv.CreateImage(cv.GetSize(cv_im),  cv.IPL_DEPTH_8U, 3)
	# cv.CvtColor(cv_im, img, cv.CV_GRAY2BGR)  
	# Affichage_watershed(cv_segmentation, img, cv_markers_8bit)
	return segmentation
	

	
def ME(imIn, imOut=-1):	
	"""
	Computes an automatic threshold image using maiximisation entropy
	"""
	import numpy as np
	if imOut == -1:
		hist = np.array(imIn, dtype = np.float32)
		print(hist)
	else:
		hist = np.array(getHistogram(imIn))
	hist = hist/float(np.max(hist))
	
	imgEntropy = 0.0
	for i in xrange(1,len(hist)-1):
		imgEntropy += i*hist[i]*np.log(i)
	argMinCE=0.0;	 minCE = 10.0**12;
	for t in xrange(len(hist)):
		#mean value of low range image
		lowValue, lowSum = 0.0, 0.0
		for i in xrange(t):
			lowValue += float(i)*hist[i];
			lowSum += hist[i]
			if lowSum>0:
				lowValue /= lowSum
		#mean value of high range image
		highValue, highSum = 0.0, 0.0
		for i in xrange(t+1,len(hist)):
			highValue += float(i)*hist[i]; highSum += hist[i]
			if highSum>0:
				highValue /= highSum
		#entropy of low range image
		lowEntropy = 0.0
		for i in xrange(t):
			lowEntropy += float(i)*hist[i]*np.log(lowValue+10**(.6));
		#entropy of high range image
		highEntropy = 0.0
		for i in xrange(t,len(hist)):
			highEntropy += float(i)*hist[i]*np.log(highValue+10**(.6));
		#Cross Entropy
		CE = imgEntropy - lowEntropy - highEntropy
		if CE<minCE:
			minCE = CE;
			argMinCE = t;
	print ("seuillage :%i"%(int(argMinCE)))
	 # Final computation
	if imOut != -1:	 
		threshold(imIn, imOut, int(argMinCE), len(hist)-1)   
	return argMinCE

def autoThreshold(imIn, imOut):
    """
    Computes an automatic threshold image using the gradient.
    This function works well with greyscale images displaying two
    highly contrasted sets.
    It produces a binary image that sort of *segment* the two sets in two.
    """
    import mambaComposed as mC 
    grad = imageMb(imIn)
    wrk = imageMb(imIn)
    level = imageMb(imIn, 1)
    # First the gradient is computed
    mC.gradient(imIn, grad)
    
    # Then the histogram
    histo = getHistogram(imIn)
    
    distri = []
    for i in range(256):
        # First no point at looking at a particular value if there is no
        # pixel in it.
        if histo[i]!=0:
        
            # for each each possible pixel value, we extract the pixels
            # in imIn with that value
            threshold(imIn, level, i, i)
            # then we compute the volume of their corresponding pixels
            # in the gradient image (normalised by the number of
            # pixels)
            mul(level, grad, wrk)
            vol = computeVolume(wrk)/histo[i]
            # The volume is added to a distribution function
            distri.append(vol)
        else:
            distri.append(0)
            
    # Finding the median of the distribution
    sd = sum(distri)
    sr = distri[0]
    threshval = 0
    while(sr<(sd/2)):
        threshval += 1
        sr += distri[threshval]
            
    # Final computation
	if imOut != -1:
		threshold(imIn, imOut, threshval, 255)
    
    return threshval
	
if __name__=="__main__":
	

	# Reading the image.
	# frame=Tk.Tk()
	path=tkFileDialog.askopenfilename(filetypes = [("Fichiers Image","*.jpeg;*.jpg;*.png;*.bmp"),("All","*")]) 		
	# frame.quit()
	# frame.destroy()
	# path="filtrage/filtrage_meanshift.jpg"
	cv_im=cv.LoadImage(path)


	cv_im_gray=RGB2L(cv_im)
	cv_im_corr_light=Correction_light(cv_im_gray,'polynomial',0)
	
	im1 = fillImageWithIplimage(cv_im_corr_light)
