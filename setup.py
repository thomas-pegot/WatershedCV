# -*- coding: iso8859-1 -*- 

import sys
from cx_Freeze import setup, Executable


base = None
if sys.platform == "win32":
    base = "Win32GUI"

executables = [
		Executable("menu1.py", base = base,  icon= "catarob.ico")
		
]
 
buildOptions = dict(
		compressed = True,
		includes = ["Analyz", "cvnumpy","findEllipse","matplotlib.pylab","MambaTools","mambaComposed","mamba","create_html","HTML","rectif","pyexiv2","watershed"],
		excludes = ['_tkagg', 'bsddb', 'curses', 'email', 'pywin.debugger',
					'pywin.debugger.dbgcon', 'pywin.dialogs',"scipy.stats"
					],					
		path = sys.path 
)
 
setup(
	name = "GranulAuto",
	version = "0.5",
	description = "Analyse automatique de sediments (OSR,Catarob)",
	options = dict(build_exe = buildOptions),
	executables = executables
)


