## Summary

### What's still-utils?
Set of scripts, which are still useful for processing of crystal diffraction data with CrystFEL.
These data are quite often produced by either serial synchrotron crystallography (SSX) or serial femtosecond crystallography (SFX). Thereby, each single crystal undergoes 0 angle rotation during diffraction, and the produced images are usually called **stil images**.

Also, still-utils aims to become a "give-me-your-data-get-HKL-in-return" package for stil diffraction image processing, but it's not there yet.

### What kind of scripts?
Scripts are written in bash and python3, and usually have somewhat expected CLI: run `script.sh -h` and inspect the output.

## Installation
You should have `git`, `python3` and `python3-pip` installed. If not and you don't have root access to your machine, please ask your systems administrator to do so. In case you have it, run:

```bash
# Debian
sudo apt-get install python3 python3-pip git
```

### Clone the repository
After doing so, please clone this repo. I prefer having all my repos in `~/github`, so here is the command:
```bash
mkdir -p ~/github/still-utils && git clone https://github.com/marinegor/still-utils.git ~/github/still-utils
```

### Install necessary python3 packages
Since you now have `pip3`, just do:

```bash
# this will only install packages for current user, not troubling others
python3 -m pip3 install --user -r requirements.txt
```

### Configure your PATH variable
Add this to your `~/.bashrc`
```bash
export PATH="$PATH:~/github/still-utils"
```
so that you won't have to type full path to the scripts every time.
