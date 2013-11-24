# -*- coding: iso8859-1 -*- 
import HTML
import numpy as np
def image(text, url, hspace=225):
	return  '''<center>%s</center><img src="%s" HSPACE=225/>'''% (text, url)


def data2html(resize, filename_current_image, filtrage, model, focale, zoom, hauteur, taille_pixel, correction_lumiere,\
	methode, filename_html, filename_current_plot = None, D50=None, D10=None, D90=None, D16=None, D25=None, D75=None, D84=None, axes_b=None, histo=None, nb_particles=None, std=None, mean=None, time_elapsed=None):	

	
	# open an HTML file to show output in a browser
	HTMLFILE = filename_html
	f = open(HTMLFILE, 'w')


	#===TITRE ====================================================================
	html_titre= '''
	  <h1>Analyz <img src="Logo.jpg" title="ENS Lyon" align="right" /> </h1>
		<p>Données granulométrique sur une image. </p>

	 '''
	print html_titre
	f.write(html_titre)
	f.write('<p>')
	print '-'*79



	html_image = image(filename_current_image,filename_current_image)
	print html_image
	f.write(html_image)
	f.write('<p>')
	print '-'*79

	#=== TABLES ===================================================================

	# 1) a simple HTML table may be built from a list of lists:

	f.write("<p><u><b>Caractéristiques Photo</b></u></p>")
	table_data = [
			['Model',  model],
			['focale (mm)', focale],
			['focale apparente (mm)', focale*zoom ],
			['zoom numérique',  zoom],
			['hauteur (cm)',  hauteur],		
			['taille pixel (pixel/mm)',  taille_pixel]		
		]

	htmlcode = HTML.table(table_data)
	print htmlcode
	f.write(htmlcode)
	f.write('<p>')
	print '-'*79

	#-------------------------------------------------------------------------------

	# 2) a header row may be specified: it will appear in bold in browsers
	f.write("<p><u><b>Paramètres de configuration</b></u></p>")
	table_data = [
			['Resize',resize],
			['Filtrage (spatial, range, echelle)', filtrage],
			['Correction lumière', correction_lumiere ],
			['Methode analyse', methode  ],
		]

	htmlcode = HTML.table(table_data)
	print htmlcode
	f.write(htmlcode)
	f.write('<p>')
	print '-'*79


	f.write("<p><u><b>Résultats</b></u></p>")
	f.write("<p> temps écoulé : %s s</p>"%(float(time_elapsed)))	
	f.write("<p> nombre particule : %s </p>"%(float(nb_particles)))
	f.write("<p><u>Tableau des percentiles</u></p>")	
	# Calcul des percentiles
		
	taille_pixel = float(taille_pixel[0])
	table_data = [
			['pixel', D10, D16, D25, D50, D75, D84, D90, mean, std ],
			['mm', D10*taille_pixel, D16*taille_pixel,  D25*taille_pixel,  D50*taille_pixel,  D75*taille_pixel,  D84*taille_pixel,   D90*taille_pixel, mean*taille_pixel, std*taille_pixel],
		]
		
	htmlcode = HTML.table(table_data, header_row=['','D10','D16','D25','D50','D75','D84','D90','moyenne', 'ecart-type'])
	print htmlcode
	f.write(htmlcode)
	f.write('<p>')
	print '-'*79	


	#Calcul des taille de Wenworth
	f.write("<p><u>Tableau classificatoion de Wenworth</u></p>")
	
	boulder, cobble, pebble, granule, sand = None, None, None, None, None
	
	
	if axes_b != None:
		axes_b = np.array(axes_b)*taille_pixel
		sand = 100.0*np.sum(axes_b < 2.0)/len(axes_b)
		granule = 100.0*np.sum((axes_b < 4.0)*(axes_b >= 2.0))/len(axes_b)
		pebble = 100.0*np.sum((axes_b < 64.0)*(axes_b >= 4.0))/len(axes_b)
		cobble = 100.0*np.sum((axes_b < 256.0)*(axes_b >= 64.0))/len(axes_b)
		boulder = 100.0*np.sum(axes_b >= 256.0)/len(axes_b)
	elif histo != None:
		norm = np.sum(histo)*1.0
		sand = 100.0*np.sum( histo[0:(2.0/taille_pixel)] )/norm
		granule = 100.0*np.sum( histo[(2.0/taille_pixel):(4.0/taille_pixel)] )/norm
		pebble = 100.0*np.sum(histo[(4.0/taille_pixel):(64.0/taille_pixel)])/norm
		cobble = 100.0*np.sum(histo[(64.0/taille_pixel):(256.0/taille_pixel)])/norm
		boulder = 100.0*np.sum(histo[(256.0/taille_pixel):])/norm
		
	table_data = [
			['Boulder (> 256 mm)', boulder],
			['Cobble(64<->256 mm)', cobble],
			['Pebble (4<->64 mm)', pebble],
			['Gravle (2<->4 mm)', granule],
			['Sand (< 2 mm)', sand ]	
		]
		
	htmlcode = HTML.table(table_data, header_row=['Classes de Wenworth', 'Pourcentage de la classe'])
	print htmlcode
	f.write(htmlcode)
	f.write('<p>')
	print '-'*79	
	
	
	
	if filename_current_plot != None:
		if histo != None:
			title_fig = "<p><u>Courbe granulométrique (et cumulée)</u></p>"
		elif axes_b != None:
			title_fig = "<p><u> Fit ellipse et courbe axes b cumulée</u></p>"
		html_image = image('Courbe granlométrique', filename_current_plot)
		print html_image
		f.write(html_image)
		f.write('<p>')
		print '-'*79

	
	htmlcode = HTML.link('Author: Thomas Pégot-Ogier', 'thomas.pegot@gmail.com')
	print htmlcode
	f.write(htmlcode)
	f.write('<p>')
	print '-'*79
	
	f.close()
	print '\nOpen the file %s in a browser to see the result.' % HTMLFILE
