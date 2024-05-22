from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="GeoGenIE",
    version="1.0.0",
    author="Bradley T. Martin, Ph.D.",
    author_email="evobio721@gmail.com",
    description="Deep learning model to predict geographic coordinates from genome-wide SNP data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/btmartin721/GeoGenIE",
    license="GPL3",
    packages=find_packages(
        exclude=[
            "tests",
            "*.tests",
            "*.tests.*",
            "tests.*",
            "simulate_data.*",
            "test_preprocessing.*",
            "*.ipynb",
            "test_outliers.*",
        ]
    ),
    install_requires=[
        "geopandas",
        "geopy",
        "imblearn",
        "jenkspy",
        "kaleido",  # Required for Optuna plots; not explicitly imported.
        "kneed",
        "matplotlib",
        "numba",
        "numpy",
        "optuna",
        "pandas",
        "plotly",  # Required for Optuna plots; not explicitly imported.
        "pynndescent",
        "pysam",
        "requests",
        "scikit-learn<=1.3.2",
        "scipy",
        "seaborn",
        "statsmodels",
        "torch",
        "xgboost",
        "pyyaml",
    ],
    setup_requires=["numpy"],
    entry_points={
        "console_scripts": [
            "geogenie=geogenie.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.11",
    include_package_data=True,
    extras_require={
        "dev": [
            "demes",
            "msprime",
            "pdf2image",
            "PyPDF2",
            "pytest",
            "snpio",
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
)
