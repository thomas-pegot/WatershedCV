# -*- coding: iso8859-1 -*- 
# from basic_units import cm
import numpy as np
from matplotlib import patches
import pylab as pl
from pylab import figure, show
import time
import pymorph
from progress import *
import Tkinter as Tk

def bounding_ellipse(pts, disp=1):
	"""
	Ellipse englobante de la forme
	"""
	
	(x,y)=pts
	max=0; ind_pt1=0; ind_pt2=0;
	# Recherche de la distance maximale dans le nuage de pts
	for i in np.arange(len(x)):
		dist=(x[i]-x)**2+(y[i]-y)**2
		curr_max=np.max(dist)
		if curr_max>max:
			max=curr_max
			# index des deux points les plus eloignees
			ind_pt1=np.lexsort((dist,dist))[-1]
			ind_pt2=i
	a=np.sqrt(max)
	
	v=[ x[ind_pt1]-x[ind_pt2], y[ind_pt1]-y[ind_pt2] ]
	# Recherche de l'axe b	
	dist_2_a=[( pl.dist_point_to_segment([x[i],y[i]],[x[ind_pt1],y[ind_pt1]],[x[ind_pt2],y[ind_pt2]])  ) for i in range(len(x))]
	b=np.max(dist_2_a)*2

	if disp==1:
		#angle
		tetha = np.arctan2(v[1],v[0])
		#centre
		cm=( (x[ind_pt1]+x[ind_pt2])*0.5, (y[ind_pt1]+y[ind_pt2])*0.5 )
		return cm, b, a , tetha
	return b

###            A TESTER   
def fit_ellipse(pts):
	(x,y)=pts
	cm=(x.mean(),y.mean())
	D=np.transpose([x**2, x*y, y**2, x, y, np.ones(len(x))])
	S=np.dot(np.transpose(D),D)
	C=np.zeros((6,6))
	C[5,5]=0; C[0,2]=2; C[1,1]=-1; C[2,0]=2
	(gevec, geval)=pl.eig( np.dot(pl.inv(S),C) ) 
	(PosR, PosC)= np.where(geval>0 & ~np.isinf(geval))
	a=gevec[:,PosC]
	
	(v,w)=pl.eig(np.array([[a[0],a[1]/2],[a[1]/2,a[2]]]))
	vect1=w[:,0]
	theta=np.arctan2(vect1[1],vect1[0])
		
	return cm, v[1], v[0], theta




def fitEllipseByMoments(bw, disp=1):
	# Ellipse center
	(x,y)=np.where(bw==True)
	cm=(x.mean(),y.mean())
	#  Momments
	M20=np.sum((x-cm[0])**2)/len(x)
	M11=np.sum((y-cm[1])*(x-cm[0]))/len(x)
	M02=np.sum((y-cm[1])**2)/len(x)

	moments=np.array([[M20,M11],[M11,M02 ]])
	
	if disp==1:
		# Param. Ellipse
		(w,v)=pl.eig(moments)
		b=4*np.sqrt(w[0]);a=4*np.sqrt(w[1]);
		tetha=np.arctan2(v[1,0],v[0,0])
		# print(a,b,tetha)
		# tetha=0.5*np.arctan2(2*M11,(M20-M02))	
		# b=4*np.sqrt(M20*np.cos(tetha)**2+2*M11*np.cos(tetha)*np.sin(tetha)+M02*np.sin(tetha)**2)
		# a=4*np.sqrt(M20*np.cos(tetha)**2-2*M11*np.cos(tetha)*np.sin(tetha)+M02*np.sin(tetha)**2)
		# print(a,b,tetha)
		return cm,a,b,tetha
	vals=pl.eigvals(moments)
	b=4*np.sqrt(np.min(vals))
	return b


def findEllipse( In, method, disp=0, pick_fig=False):
	"""
		Parameters
		-----------
		In : array_like or tuple of array depends on mode 
			Input data.
		method: string ( 'Box', 'Moment', 'all' )	
		
		Returns
		-------
		Out : list of b-axis	
	"""


	b_axis=[]; pts=[]; cmpt=0;
	labels=np.unique(In)

	#Bar progression
	bar=Tk.Tk(className='Fitting Ellipses...')
	m = Meter(bar, relief='ridge',fillcolor='grey', bd=2)
	m.pack(fill='x')
	m.set(0.0, 'Starting...')
	M=np.max(labels)
	if disp==1:
		fig = pl.figure()	
		ax = fig.add_subplot(221, aspect='auto')
		# ax.set_title('Image des labels')
		pl.imshow(In)	
		In=np.fliplr(np.transpose(In))
		img_size=np.shape(In)
		if method=='Box':	
			for label in labels[labels>0]:
				# Methode 1
				curr_object=pymorph.gradm(In==label) 
				pts.append(np.where(curr_object==True))
				ax = fig.add_subplot(222, aspect='auto')
				pl.plot(pts[cmpt][0],pts[cmpt][1],'+', markersize=6)
				cmpt+=1
				m.set(0.5*float(cmpt)/M)
			cmpt=0
			for label in labels[labels>0]:
				(cm,a,b,tetha) = bounding_ellipse(pts[cmpt])
				xcenter, ycenter =cm[0],cm[1]
				angle = 180*tetha /np.pi
				b_axis.append(np.min(np.array([b,a])))
				# Affichage
				ax = fig.add_subplot(222, aspect='auto')
				# ax.set_title('Ellipses englobantes')
				e = patches.Ellipse((xcenter, ycenter), b, a,
							 angle=angle, linewidth=2, fill=False, zorder=2)
				ax.add_artist(e)
				e.set_clip_box(ax.bbox)
				e.set_alpha(0.8)
				pl.xlim([0, img_size[0]])
				pl.ylim([0, img_size[1]])
				cmpt+=1
				m.set(0.5+0.5*float(cmpt)/M)	
		elif method=='Moments':	
			for label in labels[labels>0]:
				# Methode 1
				curr_object=In==label #pymorph.gradm(In==label) 
				pts.append(np.where(curr_object==True))
				ax = fig.add_subplot(222, aspect='auto')
				pl.plot(pts[cmpt][0],pts[cmpt][1],'+', markersize=6)
				cmpt+=1
				m.set(0.5*float(cmpt)/(M))
			cmpt=0
			for label in labels[labels>0]:
				#Methode 2
				(cm,a,b,tetha)=fitEllipseByMoments(In==label)
				xcenter, ycenter =cm[0],cm[1]
				angle = 180*tetha /np.pi
				b_axis.append(np.min(np.array([b,a])))
				##  Affichage
				ax = fig.add_subplot(222, aspect='auto')
				# ax.set_title('Ellipses de meme moments')
				e = patches.Ellipse((xcenter, ycenter), b, a,
							 angle=angle, linewidth=2, fill=False, zorder=2)
				ax.add_artist(e)
				e.set_clip_box(ax.bbox)
				e.set_alpha(0.8)
				pl.xlim([0, img_size[0]])
				pl.ylim([0, img_size[1]])
				cmpt+=1
				m.set(0.5+0.5*float(cmpt)/(M))
		ax = fig.add_subplot(223, aspect='auto')
		# ax.set_title('Fréquence cumulée axes b en pixel')
		
		pl.hist(b_axis, cumulative=True)
		pl.title('Courbe granulometrique cumulee')
		fig.show()
	else:
		if method=='Box':
			print("Method Box")
			for label in labels[labels>0]:
				#Methode 1
				contour=pymorph.gradm(In==label)
				pts=np.where(contour==True)
				b = bounding_ellipse(pts,0)
				b_axis.append(b)
				cmpt+=1
				m.set(float(cmpt)/(M))
		elif method=='Moments':
			print("Method Moments")
			for label in labels[labels>0]:
				#Methode 2		
				b=fitEllipseByMoments(In==label,0)
				b_axis.append(b)
				m.set(float(cmpt)/(M))
				cmpt+=1
	bar.destroy()
	print("->D50, Methode ellipse par "+method+":"+repr(np.median(b_axis))+" pixels")
	if pick_fig == True & disp==1:
		return b_axis, fig
	else:
		return b_axis, None

