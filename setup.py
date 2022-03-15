import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "PyAnEn", "version.py")) as fp:
    exec(fp.read())

setuptools.setup(
    name="PyAnEn",
    version=__version__,
    author="Weiming Hu",
    author_email="huweiming950714@gmail.com",
    description="The python interface to parallel Analog Ensemble",
    url="https://github.com/Weiming-Hu/PyAnEn",
    packages=setuptools.find_packages(exclude=("tests",)),
    python_requires=">=3",
    license='LICENSE',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'xarray',
        'properscoring',
        'netCDF4',
        'numpy',
        'scipy',
        'sklearn',
        'pandas',
        'dill',
    ],
)
