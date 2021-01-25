import setuptools

setuptools.setup(
    name="PyAnEn",
    version="0.0.1",
    author="Weiming Hu",
    author_email="huweiming950714@gmail.com",
    description="The python interface to parallel Analog Ensemble",
    url="https://github.com/Weiming-Hu/PyAnEn",
    packages=setuptools.find_packages(),
    python_requires=">=3",
    license='LICENSE',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'xarray',
    ],
)
