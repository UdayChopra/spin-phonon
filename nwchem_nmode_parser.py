#!/usr/bin/env python3

import numpy as np
import pickle

def ReadNmode(fname):

    class Nmode():
        pass

    nmode = Nmode() 

    #f=open(fname,'rb')
    #nmode.nEval=int(np.fromfile(f,dtype=np.float64,count=1)[0])
    #nmode.nFreq=int(np.fromfile(f,dtype=np.float64,count=1)[0])
    #nmode.nAtom=int(nmode.nFreq/3)
    #eVal=np.fromfile(f,dtype=np.float64,count=nmode.nEval)
    #data=np.fromfile(f,dtype=np.float64)
    #f.close()

    #eVec=np.reshape(data,(nmode.nFreq,nmode.nAtom,3))       
    #nmode.nVibEval=nmode.nEval - 6
    #nmode.vibEval=eVal[6:]             # Vib. frequencies
    #nmode.vibEvec=eVec[6:,:,:]         # Vib. eignenvectors

    with open(fname, 'rb') as f:
        nmode.nAtom = pickle.load(f) 
        nmode.vibEval = pickle.load(f)
        nmode.nVibEval = len(nmode.vibEval)
        nmode.vibEvec = pickle.load(f)

    return nmode
