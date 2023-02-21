Coding quality
'''''''''''''''''''''

I value software quality. Higher quality software has fewer defects, better security, and better performance, which leads to happier users who can work more effectively.
Code reviews are an effective method for improving software quality. This library is developed with several techniques, such as coding styling, low complexity, docstrings, reviews, and unit tests.
Such conventions are helpfull to improve the quality, make the code cleaner and more understandable but alos to trace future bugs, and spot syntax errors.


library
-------

The file structure of the generated package looks like:


.. code-block:: bash

    path/to/d3graph/
    ├── .editorconfig
    ├── .gitignore
    ├── .pre-commit-config.yml
    ├── .prospector.yml
    ├── CHANGELOG.rst
    ├── docs
    │   ├── conf.py
    │   ├── index.rst
    │   └── ...
    ├── LICENSE
    ├── MANIFEST.in
    ├── NOTICE
    ├── d3graph
    │   ├── __init__.py
    │   ├── __version__.py
    │   └── d3graph.py
    ├── README.md
    ├── requirements.txt
    ├── setup.cfg
    ├── setup.py
    └── tests
        ├── __init__.py
        └── test_d3graph.py


Style
-----

This library is compliant with the PEP-8 standards.
PEP stands for Python Enhancement Proposal and sets a baseline for the readability of Python code.
Each public function contains a docstring that is based on numpy standards.
    

Complexity
----------

This library has been developed by using measures that help decreasing technical debt.
Version 0.1.0 of the ``d3graph`` library scored, according the code analyzer: **VALUE**, for which values > 0 are good and 10 is a maximum score.
Developing software with low(er) technical dept may take extra development time, but has many advantages:

* Higher quality code
* easier maintanable
* Less prone to bugs and errors
* Higher security


Unit tests
----------

The use of unit tests is essential to garantee a consistent output of developed functions.
The following tests are secured using :func:`tests.test_d3graph`:

* The input are checked.
* The output values are checked and whether they are encoded properly.
* The check of whether parameters are handled correctly.





.. include:: add_bottom.add