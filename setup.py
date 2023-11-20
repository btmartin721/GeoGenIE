from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="GeoGenIE",
    version="0.1",
    description="Deep learning models to predict geographic coordinates from genome-wide SNP data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/btmartin721/GeoGenIE",
    license="GPL3",
    packages=find_packages(exclude=[]),
    install_requires=[
        "pysam",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas",
        "torch",
        "botorch",
        "optuna",
        "scipy",
        "scikit-learn",
        "tqdm",
        "plotly",
        "kaleido",
        "pyyaml",
    ],
    setup_requires=["numpy"],
)
