from setuptools import setup, find_packages

setup(
    name='bmitools',
    version='0.1.9',
    packages=find_packages(),
    include_package_data=True,
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

