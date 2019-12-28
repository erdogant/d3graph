# -*- coding: utf-8 -*-
# setup.py template made by the 'datafolder' package
# for the d3graph project.

# If you need help about packaging, read
# https://python-packaging-user-guide.readthedocs.org/en/latest/distributing.html


import sys
import pkg_resources

from setuptools import find_packages, setup

from d3graph.bootdf import Installer, DataFolderException

# write the name of the package (in this case 'mypkg'!)
MYPKG = 'd3graph'

# d3graph supports these python versions
SUPPORT = ('3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8')        # <-- ADAPT THIS

# list of data files in d3graph (just the names)
# [don't forget to include these files in MANIFEST.in!]
MYDATAFILES = ['d3.v3.js', 'd3graphscript.js', 'style.css']


# (many people get confused with the next step...)


# tell setup were these files are in your package
# (I assume that they are together with the first __init__.py)
MYRESOURCES = [pkg_resources.resource_filename(MYPKG, datafile)
               for datafile in MYDATAFILES]


# now, create the installer
installer = Installer(sys.argv)

# use the installer to check supported python versions
installer.support(SUPPORT)

# checks if there are already data files and makes a backup
installer.backup(MYPKG, files=MYDATAFILES)

# create the data folder and tell setup to put the data files there
try:
    DATAPATH = installer.data_path(MYPKG)
except DataFolderException:
    # here you can handle any exception raised with the creation
    # of the data folder, e.g., abort installation
    raise Exception('Abort installation!')
data_files = [(DATAPATH, MYRESOURCES)]

# now, setup can do his thing...
with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
     name=MYPKG,  
     version='0.1',
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     data_files=data_files,
     description="Interactive network in d3js",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/erdoganta/d3graph",
     packages=find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache 2.0 License",
         "Operating System :: OS Independent",
     ],
 )

# but we are NOT READY, in some cases the data files
# don't have the appropriate permissions and 'pip'
# overwrites all data files that have been
# previously installed (even if they have been changed!).
# By default '.conf', '.cfg', '.ini' and '.yaml' files
# are protected, you can change this by passing
# parameter 'fns', e.g. fns=('*.db','data.csv'), to 'pos_setup'.
installer.pos_setup(MYDATAFILES)
