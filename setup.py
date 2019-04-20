"""Setup autoimpute package."""

import io
import os
from setuptools import find_packages, setup
# pylint:disable=exec-used

# Package meta-data
NAME = "autoimpute"
DESCRIPTION = "Imputation Methods in Python"
URL = "https://github.com/kearnz/autoimpute"
EMAIL = "josephkearney14@gmail.com, shahidbarkat@gmail.com"
AUTHOR = "Joseph Kearney, Shahid Barkat"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = None
REQUIRED = [
    "numpy",
    "scipy",
    "pandas",
    "statsmodels",
    "xgboost",
    "scikit-learn",
    "category_encoders",
    "pymc3"
]
EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long description.
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f"\n{f.read()}"
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

# Setup specification
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/md",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6"
    ],
)
