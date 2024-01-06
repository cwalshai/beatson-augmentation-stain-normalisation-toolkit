# Beatson Augmentation and Stain Normalisation Toolkit

Welcome to the Beatson Augmentation and Stain Normalisation Toolkit.

This is a toolkit to assist with stain normalisation and augmentation of whole slide pathology images.

To install the toolkit, please clone this directory and install with pip3 install -e /path/to/repository or alternatively install directly from GitHub with pip3 install git+https://github.com/cwalshai/beatson-augmentation-stain-normalisation-toolkit

Please note the Sparse Modelling Library (SPAMs) python package is required. Documentation for SPAMs is available [here](https://pypi.org/project/spams/). To install it on Linux, you must have the GCC and G++ libraries and the libblas-dev and liblapack-dev packages installed, as well as python3-setuptools and python3-dev.

Install the necessary packages:

```bash
apt install gcc g++ libblas-dev liblapack-dev gfortran
```

Then, create a virtual environment:

```bash
apt install python3-venv
```
```bash
python3 -m venv environment_name
```
Then activate the environment and install the python3 wheel and setup tools packages.

```bash
source enviornment_name/bin/activate

pip3 install wheel setuptools
```

Then install the beast package as above.

```bash
pip3 install git+https://github.com/cwalshai/beatson-augmentation-stain-normalisation-toolkit
```

Documentation on the toolkit's use can be found here: https://cwalshai.github.io/beatson-augmentation-stain-normalisation-toolkit-docs/api.html
