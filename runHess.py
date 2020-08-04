#!/usr/bin/env python3

from Hessian import Hessian
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser(description="Uses equilibrium XYZ file to calculate the projected Hessian and write vibrational modes to .nmode file")
parser.add_argument("xyzfile", metavar="Geometry (XYZ) file")
parser.add_argument('--runtype', choices = ['dft', 'nmode'], metavar="Run type of the script: dft or nmode", )
args = parser.parse_args()

disp = [0.02]

path  = os.getcwd()+"/"

### some new comments

for i in disp:
    subdir = path+str(i)
    Path(subdir).mkdir(parents=True, exist_ok=True)
    os.chdir(subdir)
    H = Hessian(path+args.xyzfile,i)
    if args.runtype == 'dft':
        H.calcfd()
    else:
        H.dump_nmode()
