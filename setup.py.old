import re
import setuptools

# Versioning -----------------------------------------------------------------------------------------------------------
VERSIONFILE = "d3graph/__init__.py"
if getversion := re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M):
    new_version = getversion[1]
else:
    raise RuntimeError(f"Unable to find version string in {VERSIONFILE}.")

# Setup ----------------------------------------------------------------------------------------------------------------
with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()
setuptools.setup(
    install_requires=['pandas',
                      'numpy',
                      'colourmap',
                      'networkx>2',
                      'ismember',
                      'jinja2',
                      'packaging',
                      'markupsafe==2.0.1',
                      'python-louvain',
                      'datazets'],
    python_requires='>=3',
    name='d3graph',
    version=new_version,
    author="Erdogan Taskesen",
    author_email="erdogant@gmail.com",
    description="Python package to create interactive network based on d3js.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://erdogant.github.io/d3graph",
    download_url=f'https://github.com/erdogant/d3graph/archive/{new_version}.tar.gz',
    packages=setuptools.find_packages(),
    include_package_data=True,
    license_files=["LICENSE"],
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: BSD License",
                 "Operating System :: OS Independent"])
