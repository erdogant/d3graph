import setuptools
import re

# versioning ------------
VERSIONFILE="d3graph/__init__.py"
getversion = re.search( r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Setup ------------
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['pandas','numpy','colourmap','networkx>2','ismember','jinja2==2.11.3', 'sklearn', 'packaging', 'markupsafe==2.0.1'],
     python_requires='>=3',
     name='d3graph',
     version=new_version,
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Python package to create interactive network based on d3js.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://erdogant.github.io/d3graph",
	 download_url = 'https://github.com/erdogant/d3graph/archive/'+new_version+'.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     license_files=["LICENSE"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
 )