from setuptools import setup

setup(
    name='bmitools',
    version='0.1.1',
    py_modules=['bmitools'],
    install_requires=[
        'numpy',
        'tqdm',
        'scipy',
        'matplotlib',
        'pyyaml',
        'networkx',
        'scikit-learn',
        'pandas',
        'opencv-python',
        'parmap',
    ],
	long_description = """
	BMI utilities and analysis tools
	=================

	...
	
	
	.. code-block:: bash

	   pip install bmitools

	Usage
	-----

	To use Binarize2PCalcium, import the package in your Python code:

	.. code-block:: python

	   from bmitools import ... 


	""",
    long_description_content_type='text/markdown',
)

