# -*- coding: iso8859-1 -*- 
# try:
import cv2.cv as cv
# except:
	# import cv
import tkFileDialog
import numpy as np
import pylab as pl
import scipy.ndimage as ndimage
import pyexiv2
from cvnumpy import *

def extraction_laser(In, disp=False, Nb_lbl_max=4, proportion = 0.15):
	""" Extraction des laser en image de label
	
		Parameters
		-----------
		In : iplimage
			Input data.
		disp : option display image treatment
		
		Returns
		-------
		Out : ndarray shape=(n, 2) cordonnees centres des n lasers 
	"""

	## Conversion vers Lab recuperation a
	Lab = cv.CloneImage(In); cv.Zero(Lab)
	a = cv.CreateImage(cv.GetSize(In), cv.IPL_DEPTH_8U, 1)
	cv.CvtColor(In, Lab, cv.CV_BGR2Lab)
	cv.SetImageCOI(Lab,2)	
	cv.Copy(Lab,a)

	## Seuillage a au niveau laser
	BW = cv.CloneImage(a); cv.Zero(BW)
	T  = 0+128  #	Seuillage a 0+delta avec delta 128
	Nb_lbl = 0
	while Nb_lbl<Nb_lbl_max:
		cv.Threshold(a, BW, T, 255, cv.CV_THRESH_BINARY )
		A_BW = cv2array(BW)[:,:,0]	
		A_lbl, Nb_lbl = ndimage.label(A_BW)
		T-=1
	
	## Correspondre bon label au laser
	center = np.floor(np.array(A_BW.shape)/2)
	# Contrainte de proximite au centre de l image considération d une photo prise distante de plus de 37cm
	# -> corresond proportion diagonale laser sur diagonale image de 17%
	# proportion = 0.2
	diag_constraint = proportion*(np.sqrt(np.sum(np.array(A_BW.shape)**2)))
	new_lbl = 1
	# Verification contrainte
	A_center_lbl = A_lbl*0
	A_new_lbl = A_lbl

	for lbl in xrange(1,Nb_lbl_max+1):
		coord_current_lbl = np.unravel_index(pl.find(A_new_lbl==lbl),A_new_lbl.shape)
		coord_center = np.sum(coord_current_lbl, 1)/len(coord_current_lbl[0])
		diag_current_lbl = 2*np.sqrt(np.sum((coord_center-center)**2))
		if diag_current_lbl < diag_constraint:
			A_center_lbl[coord_center[0],coord_center[1]] = new_lbl
			A_new_lbl[A_new_lbl ==lbl] = new_lbl
			new_lbl+=1
		else: #contrainte proche du centre de l'image non verifiee
			A_new_lbl[A_new_lbl==lbl] = 0

	if new_lbl < 4:
		if (Nb_lbl_max>10) & (proportion<1) :
			extraction_laser(In, True, Nb_lbl_max,proportion+0.2)
		elif  proportion<1:
			extraction_laser(In, False, Nb_lbl_max+1,proportion)
						
	print new_lbl
	A_lbl = A_new_lbl
	
	
	if disp:
		A_a = cv2array(a)[:,:,0]	
	
		fig1 = pl.figure()
		pl.imshow(A_a, cmap = pl.cm.gray)
		pl.title('Composante a de Lab')
		fig1.show()

		fig2 = pl.figure()
		pl.imshow(A_lbl)
		pl.title('Image label:Seuillage a '+repr(T))
		fig2.show()
	return np.transpose(np.unravel_index(pl.find(A_center_lbl>0), A_lbl.shape))
	

def find_height(coord_laser, dim_image, focale, zoom, correction_defaut = True):
	""" Calcul hauteurs camera<->objet

		Parameters
		-----------
		In : iplimage
			Input data.
		disp : option display image treatment
		
		Returns
		-------
		Out : iplimage binary 0 and 255
	"""	
	print correction_defaut
	pts1=np.array([coord_laser[:,0].min(), coord_laser[:,1].min()])
	pts2=np.array([coord_laser[:,0].max(), coord_laser[:,1].max()])
	diag_laser_px = pl.dist(pts1, pts2)
	# Diagonale laser a 2m07 comme reference avec mesure de 109mm 	
	s = 43.0/np.sqrt(np.sum(dim_image**2)) 
	diag_laser_a_2m07 = 109
	if correction_defaut == True:
		ecart_angulaire_depuis_2m07 = 6.5/100
	else:
		ecart_angulaire_depuis_2m07 = 0.0
	hauteur_estimee = (focale*diag_laser_a_2m07/(s*diag_laser_px)) 
	ecart = (2068 - hauteur_estimee)*ecart_angulaire_depuis_2m07	
	hauteur = hauteur_estimee-ecart
	
	
	# Dimension laser metrologie avec ajustement de non parallelisme
	taille_pixel = (hauteur/focale)*(np.array([24.0/dim_image[0], 36.0/dim_image[1]]))
	
	
	return 	hauteur, taille_pixel
	
def read_metadata(path):
	# Lecture metadonnees de limage
	metadata = pyexiv2.ImageMetadata(path); metadata.read()
	
	model = metadata['Exif.Image.Model'].value
	focal = np.float(metadata['Exif.Photo.FocalLength'].value)
	zoom = np.float(metadata['Exif.Photo.DigitalZoomRatio'].value)
	focal_apparent = focal*zoom

	return model, focal_apparent, zoom
	
def Resize(In, size_choice):
	if type(In)==cv.iplimage:
		In_Resize = cv.CloneImage(In)
		if size_choice=="694 x 462":
			In_Resize = cv.CreateImage((696,462),cv.IPL_DEPTH_8U, 3)
		elif size_choice=="1392 x 924":
			In_Resize = cv.CreateImage((1392,924),cv.IPL_DEPTH_8U, 3)		
		cv.Resize(In, In_Resize)
	else:
		if size_choice=="694 x 462":
			dim = (462,696)
		elif size_choice=="1392 x 924":
			dim = (924,1392)
		if len(In.shape)==3:
			In_Resize = In[:,:,0].copy()
			np.resize(In_Resize,dim)
			stack.append(In_Resize)
			In_Resize = np.zeros(In.shape[0:2], dtype = int)
			In_Resize = In[:,:,1].copy()
			In_Resize.resize(dim)
			stack.append(In_Resize)
			In_Resize = np.zeros(In.shape[0:2], dtype = int)
			In_Resize = In[:,:,2].copy()
			In_Resize.resize(dim)
			stack.append(In_Resize)
			In_Resize = np.array(stack)
		else:
			In_Resize = In.copy()
			In_Resize.resize(dim)
	
	return In_Resize
		

def rectif(In, cadreIn, cadreOut):
	import cv2
	# Out = cv.CloneImage(In); cv.Zero(Out)
	mmat = cv.CreateMat(3,3, cv.CV_32FC1)
	cv.GetPerspectiveTransform(cadreIn , cadreOut, mmat)
	mmat = np.asarray( mmat[:,:])
	In = cv2array(In)
	
	Out = In
	size = In.shape[0:2]
	Out = cv2.warpPerspective(In, mmat, size, Out, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, 101)#, flags=cv.CV_WARP_INVERSE_MAP )	
	Out=Out.swapaxes(0,1)
	
	#mask bg
	# mask_bg = np.zeros(Out.shape[0:2], dtype=bool)
	mask_bg = (Out[:,:,0]==101)*(Out[:,:,1]==0)*(Out[:,:,2]==0)
	
	Out = array2cv(Out)

	return Out, mask_bg
		
		
if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1:
		path = sys.argv[1]
	else:
		path = tkFileDialog.askopenfilename()
		
	In = cv.LoadImage(path)
	coord_laser=extraction_laser(In)

	model, focal_apparent, zoom = read_metadata(path)
	hauteur, taille_pixel = find_height(coord_laser, np.array([In.height, In.width]), focal_apparent, zoom)
	