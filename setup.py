import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='d3graph',  
     version='0.1',
     scripts=['d3graph'] ,
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Interactive and stand alone network using d3js",
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