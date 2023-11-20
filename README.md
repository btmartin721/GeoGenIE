# GeoGenIE

Software that uses deep learning models to predict geographic coordinates from genome-wide SNPs.

## Installation

Ideally, do this in a virutal or conda environment.

In the root project directory, enter the command:

```pip install .```

### Dependencies

+ python>=3.9
+ pysam
+ matplotlib
+ seaborn
+ numpy
+ pandas
+ torch
+ optuna
+ scipy
+ scikit-learn
+ tqdm
+ plotly
+ kaleido
+ botorch

## Usage

### Configuration File

You can set all the options for input files, model parameters, etc. in the ```config.yaml``` file.

### Running the software

From project root directory:

```python scripts/run_geogenie.py --config config.yaml```

