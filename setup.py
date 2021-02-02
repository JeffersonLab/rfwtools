import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rfwtools-adamc",
    version="0.0.1",
    author="Adam Carpenter",
    author_email="adamc@jlab.org",
    description="A package for working with C100 RF waveforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JeffersonLab/rfwtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: JLab License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.6',
    install_requires=[
        "matplotlib<=3.2",
        "numpy",
        "pandas",
        "pyparsing",
        "python-dateutil",
        "requests",
        "statsmodels",
        "tzlocal",
        "PyYAML",
        "seaborn",
        "scikit-learn",
        "tqdm",
        "lttb"
    ]
)
