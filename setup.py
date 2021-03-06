from setuptools import setup
import glob

requirements = """\
numpy==1.18.0
matplotlib==3.1.2
rmsd==1.3.2
h5py==2.10.0
scipy==1.4.1
tqdm==4.19.5
filelock==3.0.12
PyYAML==5.3.1"""

scripts = glob.glob("./crystfeler/sh/*er") + glob.glob("./crystfeler/py/*")

setup(
    name="crystfeler",
    version="0.1",
    description="Module full of CLI tools to support your SSX/SFX data processing",
    author="Egor Marin",
    author_email="marin@phystech.edu",
    install_requires=requirements.split("\n"),  # external packages as dependencies
    scripts=scripts,
)
