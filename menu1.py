# -*- coding: iso8859-1 -*-
import sys, re, string, os, time
import matplotlib
import matplotlib.backends.backend_tkagg
import Tkinter as Tk
import tkFont
import tkMessageBox
from PIL import Image, ImageTk
# try:
import cv
# except:
#import cv2.cv as cv
from cvnumpy import *
from create_html import *
from rectif import *
from Analyz import *
from progress import *
from findEllipse import *
reload(sys.modules['Analyz'])
reload(sys.modules['create_html'])
reload(sys.modules['progress'])
reload(sys.modules['findEllipse'])
reload(sys.modules['rectif'])
reload(sys.modules['cvnumpy'])

from ScrolledText import ScrolledText
import threading

class Title(Tk.Frame):
    def __init__(self, master):
        Tk.Frame.__init__(self, master, relief=Tk.RIDGE, bd=2)
        l = Tk.Label(self, text=self.label, font=('-*-lucidatypewriter-medium-r-*-*-*-200-*-*-*-*-*-*', 12, 'bold'),
                    background='dark gray', foreground='white')
        l.pack(side=Tk.TOP, expand=Tk.NO, fill=Tk.X)

def Char(c): return '0.0+%d char' % c
def Options(**kw): return kw


class TextDemo(Title):
    label = 'Fichier de configuration'
    font = ('Courier', 10, 'normal')
    bold = ('Courier', 10, 'bold')
    Highlights = {
                        'global .*': Options(foreground='white'),
                        '= .*': Options(foreground='yellow'),
                        '.*=': Options(foreground='white'),
                        '#.*': Options(foreground='DarkGreen'),
                        r'\".*?\"': Options(foreground='yellow'),
                        r'\bdef\b\s.*:':Options(foreground='blue', spacing1=2),
                        r'\b(global|def|for|in|import|from|break|continue)\b':
                        Options(font=bold, foreground ='blue')
                    }
    def __init__(self, master):
        Title.__init__(self, master)
        file_name = "config.py"
        self.text = ScrolledText(self, width=80, height=20,
                                 font=self.font, background='black',#gray65
                                 spacing1=1, spacing2=1, tabs='24')
        self.text.pack(side=Tk.TOP, expand=Tk.YES, fill=Tk.BOTH)
        self.load_file(file_name)

    def load_file(self, file_name):
        self.text.delete('1.0', Tk.AtEnd())

        content = open(file_name, 'r').read()
        self.text.insert(Tk.AtEnd(), content)

        reg = re.compile('([\t ]*).*\n')
        pos = 0
        indentTags = []
        while 1:
            match = reg.search(content, pos)
            if not match: break
            indent = match.end(1)-match.start(1)
            if match.end(0)-match.start(0) == 1:
                indent = len(indentTags)
            tagb = 'Tagb%08d' % match.start(0)
            tagc = 'Tage%08d' % match.start(0)
            self.text.tag_configure(tagc, background='', relief=Tk.FLAT, borderwidth=2)
            self.text.tag_add(tagb, Char( match.start(0)), Char(match.end(0)))
            self.text.tag_bind(tagb, '<Enter>',
                               lambda e,self=self,tagc=tagc: self.Enter(tagc))
            self.text.tag_bind(tagb, '<Leave>',
                               lambda e,self=self,tagc=tagc: self.Leave(tagc))
            del indentTags[indent:]
            indentTags.extend( (indent-len(indentTags))*[None] )
            indentTags.append(tagc)
            for tag in indentTags:
                if tag:
                    self.text.tag_add(tag, Char(match.start(0)),
                                      Char(match.end(0)))
            pos = match.end(0)

        for key,kw in self.Highlights.items():
            self.text.tag_configure(key, cnf=kw)
            reg = re.compile(key)
            pos = 0
            while 1:
                match = reg.search(content, pos)
                if not match: break
                self.text.tag_add(key, Char(match.start(0)),Char(match.end(0)))
                pos = match.end(0)

    def Enter(self, tag):
        self.text.tag_raise(tag)
        self.text.tag_configure(tag, background='gray25', relief=Tk.RAISED)

    def Leave(self, tag):
        self.text.tag_configure(tag, background='', relief=Tk.FLAT)


class Calibration(Tk.Frame):
    """Curseurs pour choisir paramètre des carac. spatial, range puis echelle pyramide"""
    def __init__(self, parent, button ):
        Tk.Frame.__init__(self, parent)
        global path
        self.parent = parent
        self.button=button
        self.initialize()

    def initialize(self):
        global resize
        self.current_image, self.mask = None, None
        self.text=Tk.StringVar();
        self.correction_laser = Tk.IntVar(); self.correction_laser.set(correction_laser)
        self.size_choice=Tk.StringVar();
        Tk.Label(self, text="Resize").pack(side=Tk.TOP)
        self.size_choice.set(resize)
        self.op = Tk.OptionMenu(self, self.size_choice, '694 x 462', '1392 x 924', 'None')
        self.op.pack()
        Tk.Button(self, text="Resize", command=self.launch_resize).pack()
        Tk.Label(self, textvariable=self.text, font=("Helvetica",10)).pack(side=Tk.BOTTOM)
        Tk.Button(self, text="rectification", command=self.launch_rectif).pack()
        Tk.Checkbutton(self, text="Correction parallélisme laser", variable=self.correction_laser).pack()

    def update(self):
            self.size_choice.set(resize)

    def launch_loading(self):
        global current_image
        current_image = ChargeImage(path)
        try:
            print("charge données exif")
            self.model, self.focal_apparent, self.zoom = read_metadata(path)

        except:
            tkMessageBox.showwarning("Chargement données EXIF","Erreur de chargement des données EXIF.")
            self.model = "Unknown";self.hauteur = "Unknown";self.focal_apparent = "Unknown";self.taille_pixel= "Unknown";self.zoom="Unknown"
            pass
        if self.model == 'NIKON D700': #Si on a l'appareil photo avec les laser, alors on récupère les lasers
            try:
                print("calcul hauteur")
                self.coord_laser = extraction_laser(current_image)
                self.hauteur, self.taille_pixel= find_height(self.coord_laser, np.array([current_image.height, current_image.width]), self.focal_apparent, self.zoom, self.correction_laser.get())
                app.help.text.set("Meta-données :  \n Appareil:"+repr(self.model)+"\n Zoom numérique:"+repr(self.zoom)+\
                                "\n focale apparente:"+repr(self.focal_apparent)+" mm"\
                                "\n hauteur:"+repr(int(self.hauteur/10.))+" cm"\
                                "\n taille pixel (x,y):"+repr((round(self.taille_pixel[0],4),round(self.taille_pixel[1],4)))+" mm/pixel")
            except ValueError:
                tkMessageBox.showwarning("calcul hauteur","Erreur dans calcul hautueur et taille pixel:%s"%(ValueError) )
                openf()

    def disp(self):
        app.help.text.set("Meta-données :  \n Appareil:"+repr(self.model)+"\n Zoom numérique:"+repr(self.zoom)+\
                        "\n focale apparente:"+repr(self.focal_apparent)+" mm"\
                        "\n hauteur:"+repr(int(self.hauteur/10.))+" cm"\
                        "\n taille pixel (x,y):"+repr((round(self.taille_pixel[0],4),round(self.taille_pixel[1],4)))+" mm/pixel")

    def launch_rectif(self):
        global current_image
        #Extraction des coordonnees laser array (4,2)
        self.coord_laser = extraction_laser(current_image)
        print(self.coord_laser)
        #conversion (i,j) vers (x,y)
        rows = np.array(self.coord_laser)*0.0
        rows[:,0] = self.coord_laser[:,1]
        rows[:,1] = self.coord_laser[:,0]
        self.coord_laser = rows

        #conversion tuple
        coord_laser = tuple((tuple(self.coord_laser[0]),\
                            tuple(self.coord_laser[1]),\
                            tuple(self.coord_laser[3]),\
                            tuple(self.coord_laser[3])))
        #Calcul des coordonnees laser dans le cas orthogonal
        height_laser_pix = 0.5*74.5/self.taille_pixel[0]
        width_laser_pix = 0.5*74.5/self.taille_pixel[1]
        image_center = np.array([np.sum(self.coord_laser[:,0])/4.0,np.sum(self.coord_laser[:,1])/4.0])

        #conversion (i,j) vers (x,y)
        image_center = np.array([image_center[1], image_center[0]])
        tmp = height_laser_pix
        height_laser_pix =width_laser_pix
        width_laser_pix = tmp

        self.new_coord_laser = tuple(((image_center[0]-height_laser_pix, image_center[1]-width_laser_pix),\
                                        (image_center[0]-height_laser_pix, image_center[1]+width_laser_pix),\
                                        (image_center[0]+height_laser_pix, image_center[1]-width_laser_pix),\
                                        (image_center[0]+height_laser_pix, image_center[1]-width_laser_pix)))
        # (image_center[0]+height_laser_pix, image_center[1]+width_laser_pix),\

        print(coord_laser)
        print(self.new_coord_laser)
        print(image_center)
        #rectification
        self.current_image, self.mask = rectif(current_image, coord_laser, self.new_coord_laser)
        current_image = self.current_image
        app.canvasDemo.show_curr_img()


    def launch_resize(self):
        global current_image
        if path==None:
            openf()
        self.current_image = current_image
        # if self.old_size !=None:
        old_size = cv.GetSize(self.current_image)

        self.button["background"]="red"
        self.button.update()
        self.current_image = Resize(self.current_image, self.size_choice.get())
        if self.mask!=None:
            self.mask = Resize(self.mask, self.size_choice.get())
        current_image = self.current_image
        new_size = cv.GetSize(self.current_image)
        self.button["background"]="DarkGreen"
        if self.taille_pixel!='Unknown':
            self.taille_pixel = self.taille_pixel[0]*old_size[0]/new_size[0], self.taille_pixel[1]*old_size[1]/new_size[1]
            self.disp()
        app.canvasDemo.show_curr_img() # A MODIFIER !!


class ParamMeanShift(Tk.Frame):
    """Curseurs pour choisir paramètre des carac. spatial, range puis echelle pyramide"""
    def __init__(self, parent, button ):
        Tk.Frame.__init__(self, parent)
        global path
        self.parent = parent
        self.button=button
        self.initialize()

    def initialize(self):
        self.sp, self.sr, self.PyrScale = sp, sr, PyrScale
        self.current_image = None
        self.side1 = Tk.Scale(self,orient=Tk.HORIZONTAL,label ='Spatial criteria', from_=1, to=35,command = self.set_sp)
        self.side2 = Tk.Scale(self,orient=Tk.HORIZONTAL,label ='Range criteria', from_=1, to=35,command = self.set_sr)
        self.side3 = Tk.Scale(self,orient=Tk.HORIZONTAL,label ='Pyramid scale', from_=1, to=4,command = self.set_pyrscale)
        Tk.Button(self, text="Run", command=self.launch_msf).pack()
        self.update()
        self.side1.pack()
        self.side2.pack()
        self.side3.pack()

    def update(self):
        global sp, sr, PyyrScale
        self.sp, self.sr, self.PyrScale = sp, sr, PyrScale
        self.side1.set(sp)
        self.side2.set(sr)
        self.side3.set(PyrScale)

    def launch_msf(self):
        global current_image
        self.button["background"]="red"
        self.button.update()
        self.current_image = FiltrageMeanShift(app.mainFrame.subclasses[0].current_image, self.sp, self.sr, self.PyrScale)
        self.button["background"]="DarkGreen"
        current_image = self.current_image
        app.canvasDemo.show_curr_img() # A MODIFIER !!

    def set_pyrscale(self, p):
        self.PyrScale = int(p)
    def set_sr(self, p):
        self.sr = int(p)
    def set_sp(self, p):
        self.sp = int(p)

class ParamLight(Tk.Frame):
    """  Méthode utilisée pour la recherche de marqueur et paramètres """
    def __init__(self, parent ,button):
        Tk.Frame.__init__(self, parent)
        self.parent = parent
        self.button=button
        self.initialize()

    def initialize(self):
        self.current_image=None
        self.light_param=Tk.StringVar(); self.light_param.set(type_correction_light)
        self.lumiere=Tk.IntVar();
        Tk.Radiobutton(self, text="Méthode fréquentielle", variable=self.light_param, value='frequency', command=self.help).pack(side=Tk.TOP)
        Tk.Radiobutton(self, text="Méthode polynomiale", variable=self.light_param, value='polynomial', command=self.help).pack()
        Tk.Radiobutton(self, text="Aucune", variable=self.light_param, value='None', command=self.help).pack()
        Tk.Checkbutton(self, text="Affiche dérive éclairement", variable=self.lumiere).pack(side=Tk.LEFT)
        Tk.Button(self, text="Run", command=self.launch_LightCorrection).pack(side=Tk.RIGHT)
    def update(self):
         self.light_param.set(type_correction_light)

    def help(self):
        if self.light_param.get()=="frequency":
            app.help.text.set('Methode frequentielle:\
\n ->Recuperation composante L de l\'espace des couleurs Lab\
\n ->Determine la derive d\'éclairement (illumination) par filtrage des hautes fréquences.\
\n ->Un ajustement de contraste est ensuite réalisé\
\n Remarque: \"Affiche derive eclairement\" affiche le filtrage.')
        elif  self.light_param.get()=="polynomial":
            app.help.text.set('Methode polynomiale: \
\n ->Recuperation composante L de l\'espace des couleurs Lab\
\n ->Determine la derive d\'éclairement (illumination) par polynome ordre 3 approchant aux mieux.\
\n ->Un ajustement de contraste est ensuite réalisé\
\n Remarque: \"Affiche derive eclairement\" affiche ce polynome.')
        elif  self.light_param.get()=="None":
            app.help.text.set('Aucune: \
\n ->Recuperation composante L de l\'espace des couleurs Lab\
\n ->Un ajustement de contraste est ensuite réalisé')

    def launch_LightCorrection(self):
        global current_image
        self.button["background"]="red"
        self.button.update()
        self.current_image = correction_light( app.mainFrame.subclasses[1].current_image , self.light_param.get(),self.lumiere.get(), app.mainFrame.subclasses[0].mask)
        self.button["background"]="DarkGreen"
        # if self.light_param.get() == 'None':
            # I_8bit = cv.CreateImage(cv.GetSize(self.current_image), cv.IPL_DEPTH_8U, 1)#
            # cv.ConvertScale(self.current_image,I_8bit)
            # self.current_image  = cv.CloneImage(I_8bit)
        current_image = self.current_image
        app.canvasDemo.show_curr_img() # A MODIFIER !!

class ParamMarker(Tk.Frame):
    """  Méthode utilisée pour la recherche de marqueur et paramètres """
    def __init__(self, parent ,button):
        Tk.Frame.__init__(self, parent)
        self.parent = parent
        self.button=button
        self.initialize()

    def initialize(self):
        self.current_image, self.plt = None, None
        self.first_call_previous_object = False
        self.text = Tk.StringVar(); self.title_select_object = Tk.StringVar()
        self.text.set('Etapes extraction des objets: \n 5 methodes différentes:\n 3 methodes par segmentation \n 2 par prélèvement de granulométrie sans segmentation ')
        self.title_select_object.set("Select object")
        self.method = Tk.StringVar(); self.method.set(method_search_marker)
        self.segmented = False
        Tk.Radiobutton(self, text = "Traitement 1 et Segmentation", variable=self.method, value = "1", command=self.help).pack(side=Tk.TOP)
        Tk.Radiobutton(self, text = "Traitement 2 et fonction granulométrique", variable=self.method, value = "2", command=self.help).pack(side=Tk.TOP)
        Tk.Radiobutton(self, text = "Traitement 3 et Segmentation", variable=self.method, value = "3", command=self.help).pack(side=Tk.TOP)
        Tk.Radiobutton(self, text = "Traitement 4 (Mamba) et Segmentation", variable=self.method, value = "4", command=self.help).pack(side=Tk.TOP)
        Tk.Radiobutton(self, text = "segmentation manuelle", variable=self.method, value = "5", command=self.help).pack(side=Tk.TOP)
        Tk.Button(self, text="Run", command=self.launch_markersearch).pack(side=Tk.RIGHT)
        Tk.Button(self, textvariable=self.title_select_object, command=self.select_object).pack(side=Tk.RIGHT)
        Tk.Button(self, text= "Previous obj.", command=self.previous_object).pack(side=Tk.RIGHT)


    def select_object(self):
        global current_image
        self.lbl = None
        if self.title_select_object.get()=="Stop select!":
            self.title_select_object.set("Select object")
            app.canvasDemo.click = 0
            if np.sum(self.image_new_lbl)!=0:
                self.current_image = self.image_new_lbl
                current_image = self.current_image
        else:
            self.title_select_object.set( "Stop select!" )
            if self.segmented == True:
                app.canvasDemo.click = 1
                #initialisation
                self.image_new_lbl = self.current_image*0
                #Creation image de selectin
                self.fig = pl.figure()
                pl.imshow( self.image_new_lbl )
                pl.title( 'Particules gardees' )
                self.fig.show()
            else:
                tkMessageBox.showwarning("Select object", "Aucune image segmentée: \n Réaliser au préalable la méthode 1, 3, 4 ou 5");

    def previous_object(self):
        if self.first_call_previous_object == True:
            self.image_new_lbl -=  self.current_image*(self.current_image == self.lbl)
            pl.imshow(self.image_new_lbl)
            self.fig.show()
        self.first_call_previous_object = False

    def gestion_click(self):
        self.first_call_previous_object = True
        coord_pts = (np.floor(app.canvasDemo.coord_pts[0]*self.current_image.shape[0]/app.canvasDemo.win_size[1]) ,\
                    np.floor(app.canvasDemo.coord_pts[1]*self.current_image.shape[1]/app.canvasDemo.win_size[0]))
        self.lbl = self.current_image[coord_pts[0], coord_pts[1]]
        self.image_new_lbl +=  self.current_image*(self.current_image == self.lbl)
        pl.imshow(self.image_new_lbl)
        self.fig.show()

    def update(self):
         self.method.set(method_search_marker)

    def help(self):
        if self.method.get()=="1":
            app.help.text.set('method 1: \n Choix des marqueurs\n-> Background: Erodé ultime du contour (SKIZ)  \n \
->Foreground: Maximum régionaux \n ')
        elif  self.method.get()=="2":
            app.help.text.set('method 2: \n-> Calcul de la fonction granulométrique et de son histogramme ')
        elif  self.method.get()=="3":
            app.help.text.set('method 3: \n-> Foreground: regional maxima')
        elif  self.method.get()=="4":
            app.help.text.set('method 4: \n-> Foreground: Méthode de Mamba')
        elif  self.method.get()=="5":
            app.help.text.set('method 5: méthode manuelle d\'extraction des marqueurs \
\n -> sélectionner le centre des particules ainsi que le l\'arriere plan \
\n \tESC - sortir \
\n \tr - reset image \
\n \tw - lance segmmentation')

    def launch_markersearch(self):
        global current_image
        self.button["background"]="red"
        self.button.update()
        if self.method.get()=="2":
            global D50, D10, D90, D16, D25, D75, D84
            from MambaTools import *
            self.border= find_ROI(app.mainFrame.subclasses[2].current_image)
            self.current_image, self.histo, self.plt = granulo_func(app.mainFrame.subclasses[2].current_image, self.border)
            current_image = self.current_image
            app.mainFrame.histo = self.histo
            D50 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.5, [-1,1], 'same')==1)
            D10 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.1, [-1,1], 'same')==1)
            D90 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.9, [-1,1], 'same')==1)
            D16 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.16, [-1,1], 'same')==1)
            D25 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.25, [-1,1], 'same')==1)
            D75 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.75, [-1,1], 'same')==1)
            D84 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.84, [-1,1], 'same')==1)
        elif self.method.get()=="4":
            from MambaTools import *
            try:
                self.current_image= Segmentation(app.mainFrame.subclasses[2].current_image)
            except ValueError:
                app.help.text.set(ValueError)
            self.segmented = True
            current_image = self.current_image
        elif self.method.get()=="5":
            from watershed import *
            self.current_image, current_image = watershed_manuel(app.mainFrame.subclasses[1].current_image)
            self.segmented = True
        else:
            self.current_image, current_image= SearchMarker(app.mainFrame.subclasses[2].current_image, app.mainFrame.subclasses[1].current_image, self.method.get(),app.mainFrame.subclasses[0].mask )
            self.segmented = True
        self.button["background"]="DarkGreen"
        app.canvasDemo.show_curr_img() # A MODIFIER !!


class ParamFit(Tk.Frame):
    """  Méthode de recherche d'ellipse """
    def __init__(self, parent ,button):
        Tk.Frame.__init__(self, parent)
        self.parent = parent
        self.button=button
        self.initialize()

    def initialize(self):
        self.b=-1
        self.plt = None
        self.method=Tk.StringVar(); self.method.set(method_fit_ellispe)
        self.first= True
        self.text=Tk.StringVar();  self.disp_ellipse=Tk.IntVar()
        Tk.Radiobutton(self, text="Search Bounding Box", variable=self.method, value="Box",command=self.help).pack(side=Tk.TOP)
        Tk.Radiobutton(self, text="Method by moments ", variable=self.method, value="Moments",command=self.help).pack()
        Tk.Checkbutton(self, text="Affiche Ellipses", variable=self.disp_ellipse).pack(side=Tk.LEFT)
        Tk.Label(self, textvariable=self.text,fg="blue").pack(side=Tk.RIGHT)
        Tk.Button(self, text="Run", command=self.launch_fitting).pack(side=Tk.RIGHT)

    def update(self):
        self.method.set(method_fit_ellispe)

    def help(self):
        if self.method.get()=="Box":
            app.help.text.set('method 1: \n Retourne ellipse englobant l\'objet')
        elif  self.method.get()=="Moments":
            app.help.text.set('method 2: \n Retourne ellipse avec le même moment d\'inertie de l\'objet')
        elif  self.method.get()=="Best":
            app.help.text.set('method 3: \n Retourne l\'ellipse la plus proche au sens des moindres carrés')

    def launch_fitting(self, pick_fig=False):
        global current_image, D50, D90, D10, nb_particles, mean, std, D16, D25, D75, D84
        self.button["background"]="red"
        self.button.update()
        try:
            self.b, self.plt = findEllipse(app.mainFrame.subclasses[3].current_image,self.method.get(),self.disp_ellipse.get()|pick_fig, pick_fig)
            self.D50 = np.median(self.b)
            app.mainFrame.b = self.b
            if app.mainFrame.subclasses[0].taille_pixel!='Unknown':
                self.D50_mm = self.D50*app.mainFrame.subclasses[0].taille_pixel[0]
            D50 = np.median(self.b)
            D10 = int(np.percentile(self.b, 10))
            D90 = int(np.percentile(self.b, 90))
            D16 = int(np.percentile(self.b, 16))
            D25 = int(np.percentile(self.b, 25))
            D75 = int(np.percentile(self.b, 75))
            D84 = int(np.percentile(self.b, 84))
            nb_particles = len(self.b)
            mean = np.mean(self.b)
            std = np.std(self.b)
        except ValueError:
            print ValueError
        if app.mainFrame.subclasses[0].taille_pixel!='Unknown':
            app.help.text.set('D50 = %i %s = %i %s'%(int(self.D50),' pixels', int(self.D50_mm),' mm' ))
        else:
            app.help.text.set('D50 = %i %s '%(int(self.D50),' pixels'))
        self.button["background"]="DarkGreen"
        current_image = app.mainFrame.subclasses[3].current_image



class CanvasDemo(Title):
    label = 'Image courante'
    def __init__(self, master):
        Title.__init__(self, master)
        global current_image
        current_image = cv.LoadImage("Logo.png")
        self.panel1 = None; self.coord_pts = (-1,-1)
        self.win_size = (450,300)
        self.click = 0
        self.show_curr_img()
        current_image = None

    def onmouse(self, event):
        self.coord_pts = (event.y, event.x)
        if self.click == 1:
            app.mainFrame.subclasses[3].gestion_click()
    def select(self):
        self.panel1.bind("<Button-1>", self.onmouse)

    def show_curr_img(self):
        global current_image
        display = 1
        pil_image = None
        # Conversion de numpy.ndarray OU cv.iplimage vers PIL.Image (ImageTk.Image)
        if type(current_image) == cv.iplimage:
            if current_image.nChannels == 3:
                pil_image = Image.fromstring(
                        'RGB',
                        cv.GetSize(current_image),
                        current_image.tostring(),
                        'raw',
                        'BGR',
                        current_image.width*3,
                        0)
            elif current_image.nChannels==1:
                pil_image = Image.fromstring(
                        'L',
                        cv.GetSize(current_image),
                        current_image.tostring())
            else:
                print("Error: Format non compatible nchannels!=1 ou 3")
        elif type(current_image) == np.ndarray:
            if len(np.shape(current_image)) == 3 or len(np.shape(current_image)) == 2:
                if current_image.dtype == np.uint32:
                    im8bit = np.zeros(np.shape(current_image), dtype = np.int8)
                    im8bit += np.floor(current_image*255/(np.max(current_image)-np.min(current_image)))
                    pil_image = Image.fromarray(im8bit)
                else:
                    pil_image = Image.fromarray(current_image)
            else:
                print("Error: Format non compatible nchannels!=1 ou 3")
        elif current_image==None:
            diplay=0
        else:
            display=0
            print("type d'image gérée")
        if display:
            pil_image = pil_image.resize(self.win_size, Image.ANTIALIAS)
            tk_image = ImageTk.PhotoImage(pil_image)
            try:
                if self.panel1!=None:
                    self.panel1.destroy()
                self.panel1 = Tk.Label(self, image=tk_image,width=self.win_size[0], height=self.win_size[1])
                self.select()
                self.panel1.pack()
                self.panel1.image = tk_image
            except ValueError:
                print(ValueError)
                pass

class Help(Title):
    label = 'Informations complémentataires'
    def __init__(self, master):
        Title.__init__(self, master)
        self.parent = master
        self.initialize()
    def initialize(self):
        self.text = Tk.StringVar();
        self.info = Tk.Label(self, textvariable=self.text, fg="blue")
        self.info.pack(side=Tk.TOP)
        self.info["foreground"]="black"
        self.text.set('Astuce d\'utilisation: \n\
1) Pour une utilisation automatique sur une image, cliquer sur \"Run All\"        \n\
2) Pour une utilisatoio automatique sur plusieurs image                                \n\
        Fichier-> Create New Project... (Non implémenté !!)                           \n\
3) Pour une utilisation semi-automatique:                       \n\
        Fichier-> Open                                                              \n\
        Paramètres de configuration et suivre pas à pas les étapes.                \n\n\n\
Fichier de configuration \"config.py\":                \n\
Ce fichier permet le chargement des paramètres nécessaire à la méthode automatique.      \n\
Il est chargé lors du lancement ou en allant à \"Options->Reset parameters\".                         \n\
On peut dès lors voir les modification dans l\'espace \"Fichier de configuration\".                    \n\
Si vous souhaiter le modifier:                                              \n\
        -modifiez les valeurs dans l\'espace Paramètres de configuration.            \n\
        -Option-> Save current parameters                                             \n')


class MainFrame(Title):
    label = "Paramètres configuration"
    """ Affiche paramètres """
    def __init__(self, master):
        Title.__init__(self, master)
        self.text=Tk.StringVar()
        self.b, self.histo, self.plt = None, None, None
        self.previous_choice=-1
        self.frame = Tk.Frame(self,borderwidth=1)
        self.frame.pack(side=Tk.TOP, expand=1,fill=Tk.BOTH)
        Tk.Button(self.frame, text="Run All", command = self.runall, bg='yellow').pack(side = Tk.BOTTOM)
        self.buttons = [Tk.Button(self.frame, text="Calibration", command = lambda x = None: self.change_frame(0)),
                    Tk.Button(self.frame, text="Filtrage", command = lambda x = None: self.change_frame(1)),
                    Tk.Button(self.frame, text="Correction dérive éclairement", command = lambda x = None: self.change_frame(2)),
                    Tk.Button(self.frame, text="Recherche des Objets", command = lambda x = None: self.change_frame(3)),
                    Tk.Button(self.frame, text="Détermination granulométrie", command = lambda x = None: self.change_frame(4)) ]
        self.pack_buttons()
        # Allocation des sous classes
        self.subclasses = [ Calibration(self.frame,self.buttons[0]), ParamMeanShift(self.frame, self.buttons[1]), ParamLight(self.frame, self.buttons[2]), ParamMarker(self.frame, self.buttons[3]), ParamFit(self.frame, self.buttons[4]) ]

    def pack_buttons(self):
        self.buttons[0].pack(fill=Tk.X)
        self.buttons[1].pack(fill=Tk.X)
        self.buttons[2].pack(fill=Tk.X)
        self.buttons[3].pack(fill=Tk.X)
        self.buttons[4].pack(fill=Tk.X)

    def change_frame(self, frame_choice):
        """ Destruction du frame de la sous class courante et initialisation de la nouvelle sousclasse """
        global current_image
        if (frame_choice!=4 and self.subclasses[frame_choice].current_image!=None ):
            current_image = self.subclasses[frame_choice].current_image
            app.canvasDemo.show_curr_img()

        self.subclasses[self.previous_choice].forget()
        self.buttons[self.previous_choice]["background"] = "white"
        self.pack_buttons()
        self.buttons[frame_choice]["background"]="DarkGreen"
        self.frame.pack(side = Tk.RIGHT, expand=1,fill=Tk.BOTH)
        self.subclasses[frame_choice].pack()
        self.previous_choice = frame_choice

    def show_gif(self, imagelist):
        # extract width and height info
        photo1 = Tk.PhotoImage(file=imagelist[0])
        width1 = photo1.width()
        height1 = photo1.height()
        canvas1 = Tk.Canvas(self, width=width1, height=height1)
        canvas1.pack(side=Tk.BOTTOM)
        # loop through the series of GIFs
        for k in range(0, 20):
            print imagelist[k%5], k
            photo1 = Tk.PhotoImage(file=imagelist[k%5])
            canvas1.create_image(width1/2.0, height1/2.0, image=photo1)
            canvas1.update()
            time.sleep(0.2)
        canvas1.destroy()

    def runall(self):
        global path    , current_image, time_elapsed
        imagelist = ["emoticons/failed1.gif","emoticons/failed2.gif","emoticons/failed3.gif","emoticons/failed4.gif","emoticons/failed5.gif"]

        if path==None:
            openf()
        current_image=ChargeImage(path)
        app.canvasDemo.show_curr_img()
        start_time = time.time()

        print("Running calibration...")
        app.help.text.set('Calibration ...' )
        self.buttons[0]["background"]="red"
        self.buttons[0].update()
        try:
            self.subclasses[0].launch_loading()
            # self.subclasses[0].launch_rectif()
            self.subclasses[0].launch_resize()
            self.buttons[0]["background"]="DarkGreen"
            app.canvasDemo.show_curr_img()
        except ValueError:
            app.help.text.info["fg"]="red"
            app.help.text.set('Calibration failed.. :' +repr(ValueError))
            self.show_gif(imagelist)

        print("Running Mean-Shift Filtering...")
        app.help.text.set('Filtrage ...' )
        self.buttons[1]["background"]="red"
        self.buttons[0].update()
        try:
            self.subclasses[1].launch_msf()
            self.buttons[1]["background"]="DarkGreen"
            app.canvasDemo.show_curr_img()
        except ValueError:
            app.help.info["fg"]="red"
            app.help.text.set('Filtrage image failed :' +repr(ValueError))
            self.show_gif(imagelist)

        print("Running RGB->GL, Egalize Histogramme, Correction Light...")
        app.help.text.set('Correction, conversion en gris ...' )
        self.buttons[2]["background"]="red"
        self.buttons[2].update()
        try:
            self.subclasses[2].launch_LightCorrection()
            self.buttons[2]["background"]="DarkGreen"
            app.canvasDemo.show_curr_img()
        except ValueError:
            app.help.info["fg"]="red"
            app.help.text.set('Correction failed :' +repr(ValueError))
            self.show_gif(imagelist)

        print("Running Watershed...")
        app.help.text.set('Segmentation ...' )
        self.buttons[3]["background"]="red"
        self.buttons[3].update()
        try:
            self.subclasses[3].launch_markersearch()
            self.buttons[3]["background"]="DarkGreen"
            app.canvasDemo.show_curr_img()
            if self.subclasses[3].method.get()=='2':
                self.histo = self.subclasses[3].histo
                self.plt = self.subclasses[3].plt
                time_elapsed = int(time.time()-start_time)
                #Calcul D50
                D50 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.5, [-1,1], 'same')==1)
                D10 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.1, [-1,1], 'same')==1)
                D16 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.16, [-1,1], 'same')==1)
                D25 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.25, [-1,1], 'same')==1)
                D75 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.75, [-1,1], 'same')==1)
                D86 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.86, [-1,1], 'same')==1)
                D90 =  pl.find(np.convolve(np.cumsum(self.histo)<=np.sum(self.histo)*0.9, [-1,1], 'same')==1)
                app.help.text.set("Succesful launch.\n->time elapsed:%i %s\n\
->D50= %i %s\n\ = %i %s"%(int(time.time()-start_time),"s",int(D50), 'pixels', int(D50*self.subclasses[0].taille_pixel[0]),' mm'))

        except ValueError:
            app.help.info["fg"]="red"
            app.help.text.set('Segmentation failed :' +repr(ValueError))
            self.show_gif(imagelist)
        save_current_image()

        if self.subclasses[3].method.get()=='5':
            try:
                self.subclasses[3].select_object()
                ## Ajouter semaphore
            except ValueError:
                app.help.info["fg"]="red"
                app.help.text.set('Calcul paramètres failed :' +repr(ValueError))

        if ((method_search_marker != '2')     and (self.subclasses[3].method.get()!='2')):
            print("Searching fitting ellipses....")
            app.help.text.set('Calcul paramètres ...' )
            self.buttons[4]["background"]="red"
            self.buttons[3].update()
            try:
                app.canvasDemo.show_curr_img()
                self.subclasses[4].launch_fitting(True)
                self.plt = self.subclasses[4].plt
                self.b = self.subclasses[4].b
                self.buttons[4]["background"]="DarkGreen"
                app.help.text.set("Succesful launch.\n->time elapsed:%i %s\n\
->D50= %i %s\n\ = %i %s"%(int(time.time()-start_time),"s",int(np.median(self.b)), 'pixels', int(np.median(self.b)*self.subclasses[0].taille_pixel[0]),' mm'))
            except ValueError:
                app.help.info["fg"]="red"
                app.help.text.set('Calcul paramètres failed :' +repr(ValueError))
                self.show_gif(imagelist)
            self.buttons[0]["background"]="white"
            self.buttons[1]["background"]="white"
            self.buttons[2]["background"]="white"
            self.buttons[3]["background"]="white"
            self.buttons[4]["background"]="white"
            # self.save_data()
        app.help.info["fg"]="DarkGreen"
        time_elapsed = int(time.time()-start_time)
        save_result()

    def save_data(self):
        filename = os.path.splitext( os.path.split(path)[1] )[0]
        filename = "Test4/pAxes-B_"+filename
        np.array(self.b).tofile(filename, sep=";", format = "%s")


class Application:
    def __init__(self):
        #root.iconbitmap(default='catarob.ico')
        root.title("Analyz")

    def Go(self):
        MenuBar(root)
        self.mainFrame = MainFrame(root)
        self.canvasDemo = CanvasDemo(root)
        self.help = Help(root)
        self.textDemo = TextDemo(root)

        self.PackAll(
        [
            [[self.mainFrame, self.help],
             [self.canvasDemo,self.textDemo]]#,
        ])

        root.mainloop()

    def PackAll(self, batches):
        for batch in batches:
            b = Tk.Frame(root, bd=15, relief=Tk.FLAT)
            for row in batch:
                no_expand=0
                f = Tk.Frame(b)
                for widget in row:
                    if widget==self.canvasDemo:
                        widget.pack(in_=f, side=Tk.LEFT, expand=Tk.NO, fill=Tk.BOTH)
                        no_expand=1
                    else:
                        widget.pack(in_=f, side=Tk.LEFT, expand=Tk.YES, fill=Tk.BOTH)
                    widget.tkraise()
                if no_expand==1:
                    f.pack(side=Tk.TOP, expand=Tk.NO, fill=Tk.BOTH)
                else:
                    f.pack(side=Tk.TOP, expand=Tk.YES, fill=Tk.BOTH)
            b.pack(side=Tk.TOP, expand=Tk.YES, fill=Tk.BOTH)


class MenuBar:
    def __init__(self, master):
        # creation de la barre de menu:
        barremenu = Tk.Menu(root)
        master.config(menu=barremenu)
        policemenu=tkFont.Font(root, size=9, family='Times')
        # représentation des items du menu "Fichier" sous forme de liste:
        itemmenu=[
                  ["Open", "Ctrl+O"],
                  ["Create new project...", ""],
                  ["Save current image", "Ctrl+S"],
                  ["Save current results",""],
                  ["Quit", "Alt+X"]]

        items=self.ajusteitems( itemmenu, policemenu)

        # creation du menu "Fichier"
        fichier = Tk.Menu(barremenu, tearoff=0, font=policemenu)
        barremenu.add_cascade(label="Fichier",menu=fichier)
        fichier.add_command(label=items[0], command=openf)
        fichier.add_command(label=items[1], command=Create_new_project)
        fichier.add_command(label=items[2], command=save_current_image)
        fichier.add_command(label=items[3], command=save_result)
        fichier.add_separator()
        fichier.add_command(label=items[4], command=root.destroy)

        # Création menu execution
        itemmenu=[
                  ["Reset parameters", ""],
                  ["Save current parameters", ""],
                  ["Zoom","Alt+Z"],
                  ["Show laser", ""]]
        items=self.ajusteitems( itemmenu, policemenu)

        execution = Tk.Menu(barremenu, tearoff=0, font=policemenu)
        barremenu.add_cascade(label="Options",menu=execution)
        execution.add_command(label=items[0], command = reset)
        execution.add_command(label=items[1], command = save_current_param)
        execution.add_command(label=items[2], command = zoom)
        execution.add_command(label=items[3], command = show_laser)

        # représentation des items du menu "Aide" sous forme de liste:
        itemmenu=[
                  ["A propos", ""]
                      ]
        items=self.ajusteitems( itemmenu, policemenu)
        # creation du menu "Aide"
        aide = Tk.Menu(barremenu, tearoff=0, font=policemenu)
        barremenu.add_cascade(label="Aide",menu=aide)
        aide.add_command(label=items[0], command = self.apropos)
        # afficher le menu
        master.config(menu=barremenu)


    def ajusteitems( self,itemmenu, policemenu):
        # Calcul de la longueur maxi en pixels des items du menu fichier:
        lg=0
        for i1, i2 in itemmenu:
            lg1=policemenu.measure(i1)
            lg2=policemenu.measure(i2)
            if lg1+lg2>lg:
                lg=lg1+lg2
        esp=policemenu.measure(" ")  # = nb de pixels d'un espace
        lg=lg+2*esp  # ajout de 2 espaces

        # Ajustement des espaces pour que les commandes clavier soient calées à droite
        lch=[]
        for i1, i2 in itemmenu:
            lg1=policemenu.measure(i1)
            lg2=policemenu.measure(i2)
            n=(lg-lg1-lg2)/esp
            lch.append(i1 + " "*n + i2)
        return lch

    def quit(self): root.quit();
    def apropos(self): tkMessageBox.showinfo("A propos", "Version 0.5 by Thomas Pegot-Ogier");




def change():
    Fenetre = Tk.Tk()
    Fenetre.title('Config file')
    Entree = Tk.Entry(Fenetre)     # On définit l'objet Entry qui porte le nom Entree
    Entree.pack()               # On place "Entree"
    Fenetre.mainloop() ;

def zoom():
    if type(current_image) == cv.iplimage:
        img=cv2array(current_image)[:,:,0]
    else:
        img=current_image
    cv2.imshow('preview',img)
    cv2.setMouseCallback('preview', onmouse, img)
    cv2.waitKey()

def onmouse( event, x, y, flags, img):
    h, w = img.shape[:2]
    h1, w1 = img.shape[:2]
    x, y = 1.0*x*h/h1, 1.0*y*h/h1
    zoom = cv2.getRectSubPix(img, (800, 600), (x+0.5, y+0.5))
    cv2.imshow('zoom', zoom)


def reset():
    for i in xrange(4):
        app.mainFrame.subclasses[i].forget()
    param_by_default();
    app.mainFrame.subclasses[0].size_choice.set(resize);
    app.mainFrame.subclasses[1].sr = sr; app.mainFrame.subclasses[1].sp = sp; app.mainFrame.subclasses[1].PyrScale = PyrScale;
    app.mainFrame.subclasses[2].light_param.set(type_correction_light);
    app.mainFrame.subclasses[3].method.set(method_search_marker);
    app.mainFrame.subclasses[4].method.set(method_fit_ellispe);
    app.mainFrame.subclasses[0].update()
    app.mainFrame.subclasses[1].update()
    app.mainFrame.subclasses[2].update()
    app.mainFrame.subclasses[3].update()
    app.mainFrame.subclasses[4].update()
    tkMessageBox.showinfo("Reset Parameter", "Parameters updated by default value");

def show_laser():
    global Input
    if Input!=None:
        extraction_laser(Input, disp=True)
    else:
        tkMessageBox.showerror("show laser", "Aucune image chargée")

def save_current_image():
    global current_image, number_current_image, filename_current_image, filename_current_plot
    display = 1
    pil_image = None
    fic = os.path.basename(path)
    filename_current_image = "Traitement_"+repr(number_current_image)+fic
    # Conversion de numpy.ndarray OU cv.iplimage vers PIL.Image (ImageTk.Image)
    if type(current_image) == cv.iplimage:
        cv.SaveImage(filename_current_image, current_image)
        number_current_image+=1
    elif type(current_image) == np.ndarray:
        if len(np.shape(current_image)) == 3 :
            matplotlib.image.imsave(filename_current_image, current_image)
            number_current_image+=1
        elif len(np.shape(current_image)) == 2:
            matplotlib.image.imsave(filename_current_image, current_image, cmap = pl.cm.gray)
            number_current_image+=1
        else:
            print("Error: Format non compatible nchannels!=1 ou 3")
    elif current_image==None:
        tkMessageBox.showwarning("Save current image", "Pas d'images à sauvegarder");
    else:
        tkMessageBox.showerror("Save current image", "Format non reconnu");

    filename_current_plot = "Fig_"+repr(number_current_image)+fic



def save_current_param():
    try:
        os.rename("config.py", "_config.py")
    except ValueError:
        print(ValueError)
        tkMessageBox.showwarning("Save current parameter", "fichier de configuration config.py non existant\
        \n Création du fichier config.py...")
        os.rename("_config.py", "config.py")
    obFile = open("config.py", "a")
    output = "##\n\
#        Config File\n\
global sp, sr, PyrScale, type_correction_light, method_search_marker, method_fit_ellispe, resize, filename_project, filename_html, correction_laser\n \
\n\
# Taille resize pour acceleration calcul\n\
resize = \"%s\"\n\
\n\
# Parametres du Filrage \n\
sp, sr, PyrScale = %i, %i, %i\n\
\n\
# Methode extraction de la lumiere (\"polynomial\", \"frequency\" ou None)\n\
type_correction_light = \"%s\"\n\
\n\
# Methode de recherche des marqueurs  (\"1\",...,\"4\")\n\
method_search_marker = \"%s\"\n\
\n\
# Methode de recherche des ellipses (\"Box\", \"Moments\")\n\
method_fit_ellispe = \"%s\"    \n\
\n\
# Nom du fichier de resultat enregistre (Default: indique une option d'enregistrement)\n\
filename_html = \"%s\"\n\
\n\
# Nom du projet (en cas de creation de projet)\n\
filename_project = \"%s\" \n\
\n\
# Active ou non la correction de parallelisme  entre les lasers\n\
correction_laser = %i "%(app.mainFrame.subclasses[0].size_choice.get(), app.mainFrame.subclasses[1].sp,\
app.mainFrame.subclasses[1].sr, app.mainFrame.subclasses[1].PyrScale, app.mainFrame.subclasses[2].light_param.get(),\
app.mainFrame.subclasses[3].method.get(), app.mainFrame.subclasses[4].method.get(), filename_html, filename_project, app.mainFrame.subclasses[0].correction_laser.get())
    try:
        obFile.write(output)
        obFile.close()
    except ValueError:
        print(ValueError)
        tkMessageBox.showwarning("Save current parameter", "Impossible de sauver le fichier de configuration.\n erreur:%s"%(ValueError))
        os.rename("_config.py", "config.py")
    #On recharge le nouveau fichier de configuration dans le TextScroll
    app.textDemo.load_file("config.py")
    os.remove("_config.py")

def openf():
        global path, current_image, Input;
        path = tkFileDialog.askopenfilename(filetypes = [("Fichiers Image","*.jpeg;*.jpg;*.png;*.bmp"),("All","*")])
        current_image = cv.LoadImage(path)
        Input = cv.CloneImage(current_image)
        try:
            app.mainFrame.subclasses[0].launch_loading()
        except ValueError:
            print("erreur chargement:"+repr(ValueError))
            pass
        for i in xrange(0,4):
            app.mainFrame.subclasses[i].current_image = current_image
        app.canvasDemo.show_curr_img()

def param_by_default():
    global number_current_image, current_image, Input

    #Parametres de sortie
    global D50, mean, D10, D90, nb_particles, std, time_elapsed, D16, D25, D75, D84
    D50, mean, D10, D90, nb_particles, std, time_elapsed = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
    try:
        execfile("config.py")
    except :
        tkMessageBox.showerror("Loading config.py", "Fichier de configuration config.py manquant.\n\
Veuillez insérer config.py dans le répertoire de l'application ou le créer*. \n\n\n\
*Creation du fichiers de configuration:\n\
  -Dans la fenêtre Paramètres de configuration sélectionner les paramètres désirés\n\
  -Aller dans Options-> Save current parameters");
        global sp, sr, PyrScale, type_correction_light, method_search_marker, method_fit_ellispe, resize, filename_project, filename_html
        resize = "694 x 462";sp, sr, PyrScale = 9, 9, 1;type_correction_light = "polynomial";
        method_search_marker = "3";method_fit_ellispe = "Moments";filename_html = "Default"; filename_project = "Default" ;
        pass
    number_current_image = 0
    current_image = None

def save_result():
    global filename_current_image, filename_html, path, filename_current_plot
    if filename_html == "Default":
        fic = os.path.splitext(os.path.basename(path))[0]
        _filename_html = "Resultat_%s%s"%(fic,".html")
    else:
        _filename_html = filename_html

    if app.mainFrame.plt != None:
        app.mainFrame.plt.savefig(filename_current_plot)

    try:

        kwargs = {"model" : app.mainFrame.subclasses[0].model,
                "focale" : app.mainFrame.subclasses[0].focal_apparent/app.mainFrame.subclasses[0].zoom,
                "zoom" : app.mainFrame.subclasses[0].zoom,
                "hauteur" : int(app.mainFrame.subclasses[0].hauteur/10.0),
                "taille_pixel" : (round(app.mainFrame.subclasses[0].taille_pixel[0],4),round(app.mainFrame.subclasses[0].taille_pixel[1],4)),
                "resize" : app.mainFrame.subclasses[0].size_choice.get(),
                "filtrage" : (app.mainFrame.subclasses[1].sp, app.mainFrame.subclasses[1].sr, app.mainFrame.subclasses[1].PyrScale),
                "correction_lumiere" : app.mainFrame.subclasses[2].light_param.get(),
                "filename_current_image" : str(filename_current_image),
                "methode" : app.mainFrame.subclasses[3].method.get(),
                "filename_html" : _filename_html,
                "D50" : D50, "D10" : D10, "D90" : D90, "D16" : D16, "D25": D25, "D75": D75, "D84": D84,
                "mean" : mean,
                "std" : std,
                "nb_particles" : nb_particles,
                "time_elapsed" : time_elapsed    }

        if app.mainFrame.b != None:
            kwargs['axes_b'] = app.mainFrame.b
            kwargs["filename_current_plot"] = str(filename_current_plot)
        elif app.mainFrame.histo != None:
            kwargs['histo'] = app.mainFrame.histo
            kwargs["filename_current_plot"] = str(filename_current_plot)
    except ValueError:
        print("Erreur dans save_result():"+repr(ValueError))
    try:
        data2html(**kwargs)
    except ValueError:
        tkMessageBox.showwarning("Sauvegarde résultat", "Erreur lors de sauvegarde des données.\n erreur:%s"%(ValueError))
    return 0


def Create_new_project():
    global saisie, ent, print_filenames
    frame_project = Tk.Tk()
    print_filenames = Tk.StringVar()
    frame_project.title("Create new project")
    nom = Tk.Label(frame_project, text = "Nom du projet:")
    ent = Tk.Entry(frame_project, text = "Nom projet", width=30)
    but_open = Tk.Button(frame_project, text='Lancer sur une liste d\'image', command = open_list_filenames)

    nom.pack()
    ent.pack()
    but_open.pack()


def save_csv_file(l_filename, l_D10, l_D16, l_D25, l_D50, l_D75, l_D84, l_D90, l_taille_pixel, l_nb_particles, l_hauteur, l_mean, l_std):
    import csv
    import sys

    f = open("%s.csv"%ent.get(), 'wt')
    try:
        fieldnames = ('Image', 'D10 (pixel)', 'D16 (pixel)', 'D25 (pixel)', 'D50 (pixel)', 'D75 (pixel)', 'D84 (pixel)', 'D90 (pixel)', 'D10 (mm)', 'D16 (mm)', 'D25 (mm)', 'D50 (mm)', 'D75 (mm)', 'D84 (mm)', 'D90 (mm)',\
        'nb_particles', 'taille pixel (pixel/mm)', 'hauteur (cm)', 'mean (mm)', 'std (mm)')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        headers = dict( (n,n) for n in fieldnames )
        writer.writerow(headers)
        for i in xrange(len(l_D10)):
            writer.writerow({ 'Image': os.path.basename(l_filename[i]),
                              'D10 (pixel)':round(l_D10[i],1),
                              'D16 (pixel)':round(l_D16[i],1),
                              'D25 (pixel)':round(l_D25[i],1),
                              'D50 (pixel)':round(l_D50[i],1),
                              'D75 (pixel)':round(l_D75[i],1),
                              'D84 (pixel)':round(l_D84[i],1),
                              'D90 (pixel)':round(l_D90[i],1),
                              'D10 (mm)':round(l_D10[i]*l_taille_pixel[i],1),
                              'D16 (mm)':round(l_D16[i]*l_taille_pixel[i],1),
                              'D25 (mm)':round(l_D25[i]*l_taille_pixel[i],1),
                              'D50 (mm)':round(l_D50[i]*l_taille_pixel[i],1),
                              'D75 (mm)':round(l_D75[i]*l_taille_pixel[i],1),
                              'D84 (mm)':round(l_D84[i]*l_taille_pixel[i],1),
                              'D90 (mm)':round(l_D90[i]*l_taille_pixel[i],1),
                              'nb_particles':l_nb_particles[i],
                              'taille pixel (pixel/mm)':round(l_taille_pixel[i],1),
                              'hauteur (cm)':round(l_hauteur[i],1),
                              'mean (mm)':round(l_mean[i]*l_taille_pixel[i],1),
                              'std (mm)':round(l_std[i]*l_taille_pixel[i],1)
                              })
    finally:
        f.close()

    print open("%s.csv"%ent.get(), 'rt').read()


def open_list_filenames():
    global filenames, print_filenames
    filenames = string2list( tkFileDialog.askopenfilenames() )
    print_filenames.set(filenames)
    print("Liste d\'images:\n %s"%print_filenames.get())
    launch_project()

def launch_project():
    global filenames, ent
    os.makedirs(ent.get())
    os.chdir(ent.get())

    l_D10, l_D50, l_D90, l_D16, l_D25, l_D75, l_D84 = [], [], [], [], [], [], []
    l_taille_pixel, l_hauteur, l_nb_particles, l_mean, l_std = [], [], [], [], []
    l_filenames = []

    if filenames != ['']:
        global path, current_image
        for i,path in enumerate(filenames):
            # global path, current_image
            current_image = cv.LoadImage(path)
            app.mainFrame.runall()
            print( "Nombre Images Restantes: %i"%int(len(filenames)-i) )
            l_D10.append(D10); l_D50.append(D50); l_D90.append(D90);
            l_D16.append(D16); l_D25.append(D25); l_D75.append(D75); l_D84.append(D84);
            l_taille_pixel.append(app.mainFrame.subclasses[0].taille_pixel[0]);
            l_hauteur.append(int(app.mainFrame.subclasses[0].hauteur/10.0)); l_nb_particles.append(nb_particles);
            l_mean.append(mean); l_std.append(std);
            save_csv_file(filenames, l_D10, l_D16, l_D25, l_D50, l_D75, l_D84, l_D90, l_taille_pixel, l_nb_particles, l_hauteur, l_mean, l_std)
    else:
        return 0

def string2list(input_string):
    input_string = input_string.lstrip('{')
    input_string = input_string.rstrip('}')
    output = input_string.split('} {')
    return output

if __name__ == '__main__':
    #   Initialisation:
    # Définition des paramètres par défauts définis
    param_by_default()

    #   Menu
    path=None
    root = Tk.Tk()
    root.geometry("%dx%d%+d%+d" % (800,700,100,100))

    app=Application()
    app.Go()
    # chaine= Saisie.askstring(title="", prompt="")
# entier= Saisie.askinteger(title="", prompt="")
# decimal= Saisie.askfloat(title="", prompt="")
