import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['pandas','numpy','seaborn','networkx>2'],
     python_requires='>=3.6',
     name='d3graph',  
     version='0.1.1',
#     version=versioneer.get_version(),    # VERSION CONTROL
#     cmdclass=versioneer.get_cmdclass(),  # VERSION CONTROL
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Interactive network in d3js",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/erdoganta/d3graph",
	 download_url = 'https://github.com/erdoganta/d3graph/archive/0.1.1.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
     ],
 )