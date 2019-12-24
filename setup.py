import setuptools
#MYPKG = 'd3graph'

#MYDATAFILES = ['d3.v3.js', 'd3graphscript.js', 'style.css']

# tell setup were these files are in your package
# (I assume that they are together with the first __init__.py)
#MYRESOURCES = [pkg_resources.resource_filename(MYPKG, datafile)
#               for datafile in MYDATAFILES]

#data_files = [(DATAPATH, MYRESOURCES)]

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='d3graph',  
     version='0.1',
     scripts=['d3graph_bash'],
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Interactive network in d3js",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/erdoganta/d3graph",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache 2.0 License",
         "Operating System :: OS Independent",
     ],
 )