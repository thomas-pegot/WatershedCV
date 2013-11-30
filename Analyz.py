# -*- coding: iso8859-1 -*-
"""
:mod: 'Analyz'
=================================


.. module::  Analyz
    :platform: Unix, Windows
    :synopsis: Traitement granulométrique (filtrage, segmentation, fit)

.. moduleauthor:: Thomas Pegot <thomas.pegot@gmail.com>
.. date_day:: '2013-11-24'

"""
import matplotlib.backends.backend_tkagg
import pymorph
from scipy import ndimage
from cvnumpy import *
import numpy as np
import pylab as pl
import cv2.cv as cv


def correction_light(I, method, show_light, mask=None):
    """Corrige la derive eclairement

    :I: array_like ou iplimage
    :method: 'polynomial' or 'frequency'
    :show_light: option affiche correction (true|false)
    :mask: array de zone non interet
    :returns: iplimage 32bit

    """
    from progress import *
    import Tkinter
    if type(I) == cv.iplimage:
        if I.nChannels == 3:
            if method == 'None':
                I = RGB2L(I)
                I = cv2array(I)[:, :, 0]
                I = pymorph.hmin(I, 15, pymorph.sedisk(3))
                I = array2cv(I)
                cv.EqualizeHist(I, I)
                return I
            I = RGB2L(I)
            I_32bit = cv.CreateImage(cv.GetSize(I), cv.IPL_DEPTH_32F, 1)
            cv.ConvertScale(I, I_32bit, 3000.0, 0.0)
            I = cv.CloneImage(I_32bit)
        I = cv2array(I)[:, :, 0]
    elif len(I.shape) == 3:
        I = (I[:, :, 0] + I[:, :, 0] + I[:, :, 0])\
            / 3.0  # A modifier: non utiliser dans notre cas
    elif method == 'None':
        I = array2cv(I)
        cv.EqualizeHist(I, I)
        return I

    I = np.log(I + 10 ** (-6))
    (H, W) = np.shape(I)
    I_out = I * 0 + 10 ** (-6)
    if method == 'polynomial':
        ## I = M.A avec A coeff. du polynome
        I_flat = I.flatten()
        degree = 3
        print("modification degree 3")
        #degree du polynome
        nb_coeff = (degree + 1) * (degree + 2) / 2  # nombre coefficient
        [yy, xx] = np.meshgrid(np.arange(W, dtype=np.float64),
                               np.arange(H, dtype=np.float64))
        if mask is not None:
            xx[mask] = 0
            yy[mask] = 0
        # Creation de M
        try:
            M = np.zeros((H * W, nb_coeff), dtype=np.float64)
        except MemoryError:
            print MemoryError
            return MemoryError
        i, j = 0, 0  # i,j degree de x,y
        #Bar progression
        bar = Tkinter.Tk(className='Correcting Light...')
        m = Meter(bar, relief='ridge', bd=3)
        m.pack(fill='x')
        m.set(0.0, 'Starting correction...')
        for col in np.arange(nb_coeff):
            M[:, col] = (xx.flatten() ** i) * (yy.flatten() ** j)
            i += 1
            m.set(0.5 * float(col) / (nb_coeff - 1))
            if i + j == degree + 1:
                i = 0
                j += 1

        # Resolution au sens des moindres carree: pseudo-inverse
        try:
            M = pl.pinv(M)
            A = np.dot(M, I_flat)
        except ValueError:
            return ValueError
        # Calcul de la surface
        i, j = 0, 0
        surface = np.zeros((H, W), dtype=np.float64)
        for cmpt in np.arange(nb_coeff):
            surface += A[cmpt] * (xx ** i) * (yy ** j)  # forme quadratique
            i += 1
            m.set(0.5 + 0.5 * float(cmpt) / (nb_coeff - 1))
            if i + j == degree + 1:
                i = 0
                j += 1
        bar.destroy()
        I_out = np.exp(I / surface)
        light = surface
    elif method == 'frequency':
        Rx, Ry = 2, 2
        # zero padding
        N = [H, W]
        filtre = np.zeros((N[1], N[0]))
        centre_x = round(N[0] / 2)
        centre_y = round(N[1] / 2)
        print("FFT2D...")
        I_fourier = pl.fftshift(pl.fft2(I, N))

        # Gaussian filter
        [xx, yy] = np.meshgrid(np.arange(N[0], dtype=np.float),
                               np.arange(N[1], dtype=np.float))
        filtre = np.exp(-2 * ((xx - centre_x) ** 2 + (yy - centre_y) ** 2) /
                        (Rx ** 2 + Ry ** 2))
        filtre = pl.transpose(filtre)
        I_fourier = I_fourier * filtre
        print("IFFT2D...")
        I_out = (np.abs(pl.ifft2(pl.ifftshift(I_fourier), N)))[0:H, 0:W]
        light = I_out
        I_out = np.exp(I / I_out)
    else:
        light = I * 0
        I_out = I
    # Display Light
    if show_light:
        light = ((light - light.min()) * 3000.0 /
                 light.max()).astype('float32')
        light = array2cv(light)
        fig = pl.figure()
        pl.imshow(light)
        fig.show()

    I_out = (I_out - I_out.min()) * 3000.0 / I_out.max()
    I_out = I_out.astype('uint8')

    #chapeau haut de forme
    I_out = pymorph.hmin(I_out, 25, pymorph.sedisk(3))
    #Conversion en iplimage et ajustement contraste
    gr = array2cv(I_out)
    cv.EqualizeHist(gr, gr)
    return gr


def ulterode(bw, size_se):
    """Erode ultime

    :bw: image booleen
    :size_se: taille elt structurant
    :returns: iplimage

    """

    SE = cv.CreateStructuringElementEx(size_se, size_se, np.int(size_se / 2),
                                       np.int(size_se / 2), cv.CV_SHAPE_CROSS)
    tmp = cv.CloneImage(bw)
    Erode = cv.CloneImage(bw)
    cv.Zero(Erode)
    Open = cv.CloneImage(Erode)
    Residu = cv.CloneImage(Erode)
    UltErode = cv.CloneImage(Erode)

    while(cv.Sum(tmp)[0] > 10):
        cv.MorphologyEx(tmp, Open, Erode, SE, cv.CV_MOP_OPEN)
        cv.MorphologyEx(tmp, Residu, Erode, SE, cv.CV_MOP_TOPHAT)
        cv.Or(Residu, UltErode, UltErode)
        cv.Erode(tmp, tmp, SE)
    return UltErode


def squel(cvBW):
    """ Calcul squelette image binaire par carte des distance
    """
    cvBW32 = cv.CreateImage(cv.GetSize(cvBW), cv.IPL_DEPTH_32F, 1)
    cv.ConvertScale(cvBW, cvBW32)
    cv.DistTransform(cvBW, cvBW32)
    cv.ConvertScale(cvBW32, cvBW)

    cv.Sobel(cvBW, cvBW32, 1, 1, 5)
    cv.ConvertScale(cvBW32, cvBW)
    cv.Threshold(cvBW, cvBW, 1, 255, cv.CV_THRESH_BINARY | cv.CV_THRESH_OTSU)

    return cvBW


def ChargeImage(path):
    img = cv.LoadImage(path, cv.CV_LOAD_IMAGE_COLOR)
    return img


def FiltrageMeanShift(img, sp, sr, PyrScale):
    ##  Filtre MeanShift denoising
    print("filtrage meanshift...")
    img_filtree = cv.CloneImage(img)
    cv.PyrMeanShiftFiltering(img, img_filtree, sp, sr, PyrScale)
    return img_filtree


def RGB2L(img_filtree):
    gray = cv.CreateImage(cv.GetSize(img_filtree), img_filtree.depth, 1)
    tmp = cv.CloneImage(img_filtree)
    cv.CvtColor(img_filtree, tmp, cv.CV_BGR2HSV)
    cv.SetImageCOI(tmp, 3)
    cv.Copy(tmp, gray)
    return gray


def find_ROI(In):
    Ain = cv2array(In)[:, :, 0]
    T = cv.CloneImage(In)
    cv.Threshold(In, T, 1, 255, cv.CV_THRESH_OTSU | cv.CV_THRESH_BINARY)
    I = cv2array(T)[:, :, 0]
    I = I > 0
    # Cadre blanc
    M = np.ones(np.shape(I))
    M[1: np.shape(I)[0] - 1, 1: np.shape(I)[1] - 1] = 0
    # Enleve bord
    I = pymorph.erode(pymorph.erode(pymorph.erode(I)))
    M2 = M * I > 0
    M1 = M2 * 0
    while abs(np.sum(M1 - M2)) > 0.1:
        M1 = M2
        M2 = pymorph.dilate(M2)
        M2 = M2 * I
    M2 = pymorph.dilate(pymorph.dilate(pymorph.dilate(M2)))
    return M2 * Ain


def Detection_contour(In):
    ##         Filtre de Canny sur image entree
    Il = cv.CloneImage(In)
    cv.Canny(In, Il, 1, 255, 3)
    tmp = cv.CreateImage(cv.GetSize(In), cv.IPL_DEPTH_32F, 6)
    cv.CornerEigenValsAndVecs(In, tmp, 5, 7)
    grad = cv.CreateImage(cv.GetSize(In), cv.IPL_DEPTH_32F, 1)
    cv.SetImageCOI(tmp, 1)
    cv.Copy(tmp, grad)
    Id = cv.CreateImage(cv.GetSize(In), cv.IPL_DEPTH_8U, 1)
    cv.ConvertScale(grad, Id)
    cv.Threshold(Id, Id, 1, 255, cv.CV_THRESH_OTSU | cv.CV_THRESH_BINARY)
    StrEl = cv.CreateStructuringElementEx(3, 3, 1, 1, cv.CV_SHAPE_ELLIPSE)
    S = cv.CloneImage(Id)
    test = False
    while(test == 0):
        Iold = cv.CloneImage(Id)
        cv.And(Id, Il, Id)
        cv.Dilate(Id, Id, StrEl)
        cv.AbsDiff(Iold, Id, S)
        test = cv.Sum(S)[0] < 0.1
    cv.Erode(Id, Id, StrEl)
    g = cv.CloneImage(Id)
    cv.Threshold(g, g, 1, 255, cv.CV_THRESH_OTSU | cv.CV_THRESH_BINARY)
    return g


def Label_Edge(Edge):
    A_Edge = cv2array(Edge)[:, :, 0] > 0
    A_lbl, _ = ndimage.label(A_Edge)
    return A_lbl


def SearchMarker(In, image_filtree, marker_param, mask):
    """Algorithme principale de recherche de marqueur et segmentation

    :In: image d'entree
    :image_filtree: image filtree par MeanShift
    :marker_param: choix de la methode de recherche (1,2 ou 3)
    :mask: masque de non interet
    :returns: image label, image pour affichee

    """
    if image_filtree.nChannels == 1:
        tmp = cv.CreateImage(cv.GetSize(image_filtree), cv.IPL_DEPTH_8U, 3)
        cv.CvtColor(image_filtree, tmp, cv.CV_GRAY2BGR)
        image_filtree = cv.CloneImage(tmp)
        del tmp

    fg = cv.CloneImage(In)
    objets = None

    # Image distance entre de watershed
    markers = cv.CreateImage(cv.GetSize(In), cv.IPL_DEPTH_32S, 1)
    img = cv.CloneImage(image_filtree)
    cv.Zero(img)

    edge = Detection_contour(In)
    cont = cv2array(edge)[:, :, 0]

    cv.Not(edge, edge)
    dist_map = cv.CreateImage(cv.GetSize(In), cv.IPL_DEPTH_32F, 1)
    cv.DistTransform(edge, dist_map, cv.CV_DIST_L2, cv.CV_DIST_MASK_5)

    dist_map_8bit = cv.CloneImage(edge)
    cv.Zero(dist_map_8bit)
    cv.ConvertScale(dist_map, dist_map, 3000.0, 0.0)
    cv.Pow(dist_map, dist_map, 0.3)
    cv.ConvertScale(dist_map, dist_map_8bit)

    cv.CvtColor(dist_map_8bit, img, cv.CV_GRAY2BGR)  #
    cv.AddWeighted(image_filtree, 0.3, img, 0.7, 1, img)    #
    cv.CvtColor(img, dist_map_8bit, cv.CV_BGR2GRAY)  #

    # Foreground by regional maxima: detection marqueurs
    if marker_param == "1" or marker_param == "3":
        print("Recherche max. regionaux...")
        I = cv2array(dist_map_8bit)[:, :, 0]
        If = ndimage.gaussian_filter(I, 5)
        rmax = pymorph.regmax(If)
        rmax = pymorph.close_holes(rmax) * 255
        #import ipdb;ipdb.set_trace()
        bool_fg = array2cv(rmax.astype(np.uint8))
        cv.ConvertScale(bool_fg, fg)

    if marker_param == "1":
        print("Recherche squelette...")
        from mamba import *
        from mambaComposed import *
        from MambaTools import *
        percent_edge = np.sum(cont) / (edge.width * edge.height)
        initial_shape = (In.height, In.width)
        im1 = fillImageWithIplimage(In)
        imWrk1 = imageMb(im1)
        blobsMarkers = imageMb(im1, 1)
        imWrk3 = imageMb(im1, 32)
        #backgroundMarker = imageMb(im1, 1)
        #finalMarkers = imageMb(im1, 1)
        print("taille se %s" % int(15.0 * 6 / percent_edge + 1))
        if In.height < 700:
            alternateFilter(im1, imWrk1,
                            int(15.0 * 6 / percent_edge) + 1, True)
        elif In.height < 1400:
            alternateFilter(im1, imWrk1,
                            int(30.0 * 6 / percent_edge) + 1, True)
        else:
            alternateFilter(im1, imWrk1,
                            int(60.0 * 6 / percent_edge) + 1, True)
        minima(imWrk1, blobsMarkers)
        thinD(blobsMarkers, blobsMarkers)
        nbStones = label(blobsMarkers, imWrk3)
        bg_array = getArrayFromImage(imWrk3)[
            0: initial_shape[0],
            0: initial_shape[1]]
        tmp_array = (bg_array > 0) * 255
        bg_ = array2cv(tmp_array.astype(np.uint8))
        bg = cv.CloneImage(dist_map_8bit)
        cv.ConvertScale(bg_, bg)
        cv.Or(fg, bg, fg)
    cv.ConvertScale(fg, markers)

    # Watershed
    print("Watershed...")

    storage = cv.CreateMemStorage(0)
    contours = cv.FindContours(fg, storage,
                               cv.CV_RETR_CCOMP,
                               cv.CV_CHAIN_APPROX_SIMPLE)

    def contour_iterator(contour):
        while contour:
            yield contour
            contour = contour.h_next()
    cv.Zero(markers)
    comp_count = 0
    for c in contour_iterator(contours):
        cv.DrawContours(markers,
                        c,
                        cv.ScalarAll(comp_count + 1),
                        cv.ScalarAll(comp_count + 1),
                        -1,
                        -1,
                        8)
        comp_count += 1

    cv.Watershed(img, markers)

    if img.nChannels == 3:
        cv.CvtColor(In, img, cv.CV_GRAY2BGR)
    elif img.nChannels == 1:
        img = cv.CloneImage(image_filtree)
        cv.CvtColor(In, img, cv.CV_GRAY2BGR)
    else:
        print("nChannels >< 3 or 1")
    cv.CvtColor(fg, img, cv.CV_GRAY2BGR)
    cv.CvtColor(In, img, cv.CV_GRAY2BGR)

    #bug
    wshed = Affichage_watershed(markers, img, fg)

    wshed_lbl = cv2array(markers)[:, :, 0]

    if marker_param == "1" or marker_param == "2":
        objets = cv2array(bg)[:, :, 0]
        objets = objets > 1
        print("Keep objects not in background ...")
        cmpt = 0
        label_map = wshed_lbl * 0
        for label in np.unique(wshed_lbl):
            if np.sum(objets * (wshed_lbl == label)) > 0:
                label_map[wshed_lbl == label] = 0
            else:
                cmpt += 1
                label_map[wshed_lbl == label] = cmpt
    else:
        label_map = wshed_lbl
    label_map = Enleve_bord(label_map, mask)
    return label_map, wshed


def Enleve_bord(im_lbl, mask=None):
    im_lbl = im_lbl * (im_lbl > 0)
    cadre = np.ones(im_lbl.shape)
    cadre[2: (im_lbl.shape[0] - 2), 2: (im_lbl.shape[1] - 2)] = 0
    if mask is not None:
        import pymorph
        mask = pymorph.dilate(mask)
        cadre += mask > 0
    print(cadre)
    lbl_bord = np.unique(cadre * im_lbl)
    for i in lbl_bord:
        im_lbl[im_lbl == i] = 0
    return im_lbl


def indice_circularite(label_map, seuil):
    """Fait un test de circularite sur chaque objet de l'image

    :label_map: image des label
    :seuil: seuil de circularite
    :returns: image verifiant la circularite C > seuil

    """
    for label in np.unique(label_map):
        current_label = (label_map == label)
        A = np.sum(current_label)
        edge_current_label = pymorph.gradm(current_label)
        P = np.sum(edge_current_label)
        if P > 0:
            C = 4 * np.pi * A / (P ** 2)
        else:
            C = 0
        if C < seuil:
            label_map[label_map == label] = 0
    return label_map


def Affichage_watershed(watershed_32bit, img, markers):
    """Affichage de la segmentation avec effet de chevauchement

    :watershed_32bit: iplimage 32bit
    :img: image d'entree
    :markers: marqueurs
    :returns: iplimage

    """
    rng = cv.RNG(-1)
    wshed = cv.CreateImage(cv.GetSize(watershed_32bit), cv.IPL_DEPTH_8U, 3)
    cv.Zero(wshed)
    # markers = cv.CloneImage(watershed_32bit); cv.Zero(markers)
    storage = cv.CreateMemStorage(0)
    contours = cv.FindContours(markers, storage,
                               cv.CV_RETR_CCOMP,
                               cv.CV_CHAIN_APPROX_SIMPLE)

    def contour_iterator(contour):
        while contour:
            yield contour
            contour = contour.h_next()

    comp_count = 0
    for c in contour_iterator(contours):
        cv.DrawContours(markers, c, cv.ScalarAll(comp_count + 1),
                        cv.ScalarAll(comp_count + 1), -1, -1, 8)
        comp_count += 1

    markers = cv.CloneImage(watershed_32bit)
    cv.Set(wshed, cv.ScalarAll(255))
    # paint the watershed image
    color_tab = [(cv.RandInt(rng) % 180 +
                  50, cv.RandInt(rng) % 180 +
                  50, cv.RandInt(rng) % 180 + 50)
                 for i in range(comp_count)]
    for j in xrange(markers.height):
        for i in xrange(markers.width):
            idx = markers[j, i]
            if idx != -1:
                wshed[j, i] = color_tab[int(idx - 1)]

    cv.AddWeighted(wshed, 0.35, img, 0.65, 1, wshed)
    return wshed
