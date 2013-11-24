##
#        Config File
global sp, sr, PyrScale, type_correction_light, method_search_marker, method_fit_ellispe, resize, filename_project, filename_html, correction_laser
 
# Taille resize pour acceleration calcul
resize = "694 x 462"

# Parametres du Filrage 
sp, sr, PyrScale = 12, 12, 1

# Methode extraction de la lumiere ("polynomial", "frequency" ou None)
type_correction_light = "polynomial"

# Methode de recherche des marqueurs  ("1",...,"4")
method_search_marker = "1"

# Methode de recherche des ellipses ("Box", "Moments")
method_fit_ellispe = "Moments"    

# Nom du fichier de resultat enregistre (Default: indique une option d'enregistrement)
filename_html = "Default"

# Nom du projet (en cas de creation de projet)
filename_project = "Default" 

# Active ou non la correction de parallelisme  entre les lasers
correction_laser = 1 