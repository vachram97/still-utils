from setuptools import setup
import glob

setup(name="runner", scripts=["src/runner"])
# pip3 install .

requirements = """\
numpy==1.18.0
matplotlib==3.1.2
rmsd==1.3.2
h5py==2.10.0
scipy==1.4.1
tqdm==4.19.5
PyYAML==5.3.1"""

setup(
    name="crystfeler",
    version="0.1",
    description="Module full of CLI tools to support your SSX/SFX data processing",
    author="Egor Marin",
    author_email="marin@phystech.edu",
    packages=["crystfeler"],  # same as name
    install_requires=requirements.split("\n"),  # external packages as dependencies
    scripts=glob.glob("./src/crystfeler/sh/*er") + glob.glob("./src/crystfeler/py/*"),
)
