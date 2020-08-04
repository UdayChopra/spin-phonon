#!/usr/bin/env python3

import numpy as np
import pickle
import os, fnmatch, subprocess
from math import ceil
from ase import Atoms
from ase.vibrations import Vibrations
from ase.io import read
from ase.units import Hartree, Bohr

class Hessian:

    def __init__(self,eqxyz,fd_size):
        self.atomobj = read(eqxyz) 
        self.vibobj  = Vibrations(self.atomobj,delta=fd_size)
        self.natoms  = self.atomobj.get_number_of_atoms()
        self.nmodes  = 3*self.natoms
        self.nlines  = ceil(self.nmodes/4)

    def projectout(self,t,r):
        d = np.vstack((t,r))
        q,r = np.linalg.qr(d.transpose())   # q's are a set of Schmidt-orthogonalized vectors
        q = q.transpose()
        q = np.matrix(q)
        I = np.diag(np.ones((self.nmodes)))
        W = np.dot(q.transpose(),q)
        P = I - W
        return P
    
    def eigint(self,fcmw,P):
        indices = range(self.natoms)
        ones = np.ones((self.natoms))
        im = np.repeat(ones[indices]**-0.5, 3)
        #im = np.repeat(self.atomobj.get_masses()[indices]**-0.5, 3)
        fc_int = np.dot(P.transpose(),np.dot(fcmw,P))
        val, vec = np.linalg.eigh(fc_int)           # vectors are returned in mass-weighted cartesian coordinates
        # Line below has been commented out because we're testing with mass weighted coordinates, so transformation to cartesian-coordinates is skipped.
        vec = np.array([ np.multiply(vec.transpose()[i],im) for i in range(self.nmodes) ], dtype=np.float64)   # vectors in cartesian-coordinates
        val = np.sqrt(val,dtype = np.complex128)       # Energies in eV
        return val, vec
    
    def projections(self):
        self.atomobj.translate(-self.atomobj.get_center_of_mass())
        pos = self.atomobj.get_positions()
        
        moi = np.array([ [np.sum(self.atomobj.get_masses()*(pos[:,i-2]**2 + pos[:,i-1]**2)) if i==j else -np.sum(self.atomobj.get_masses()*pos[:,i]*pos[:,j]) for j in range(3)] for i in range(3)])
        I, X = np.linalg.eigh(moi)
        
        indices = range(self.natoms)
        im = np.repeat(self.atomobj.get_masses()[indices]**-0.5, 3)
        
        self.vibobj.read()
        evec = []
        for i in range(self.nmodes):
            evec.append(self.vibobj.get_mode(i))    # here in cartesian coordinates
        evec = np.reshape(evec,(self.nmodes,self.nmodes))
        mwevec = np.array([evec[i]/im for i in range(self.nmodes)]).transpose()  # mass-weigted cartesian coordinates
        
        e = self.vibobj.get_energies()
        E = e**2
        fc = np.diag(E)
        fc_mw = np.dot(mwevec,np.dot(fc,np.linalg.inv(mwevec)))
        
        np.allclose(fc_mw,fc_mw.transpose())
        
        v =  np.array([[1,0,0],[0,1,0],[0,0,1]])
        d1 = np.array([ (np.matrix(np.sqrt(self.atomobj.get_masses())).getH()*np.matrix(v[i])) for i in range(3) ] )
        d1 = np.reshape(d1,(3,self.nmodes))
        d2 = np.array([np.cross(pos,X[:,i]) for i in range(3)])
        d2 = np.reshape(d2,(3,self.nmodes))
        d2 = d2/im
        proj = self.projectout(d1,d2)
        val, vec = self.eigint(fc_mw,proj)
        return val, vec
    
    def calcfd(self):
        outDir = os.getcwd()
        for dispName, atoms in self.vibobj.iterdisplace(inplace=True):      # this method creates generator for creating +/- finite-displacement XYZ files 
            xyzfilename = dispName+'.xyz'
            atoms.write(xyzfilename)
            subprocess.call(["../gen_tzvp_hess.sh",xyzfilename])
            #for f in os.listdir(outDir):
            #    if fnmatch.fnmatch(f,dispName+"*.nw"):
            #        #os.system("%s Q %s" %(" sub=y n=8 ", f)) # quasar/bigbang
            #        os.system("%s Q %s" %("nn=2 sub=y", f)) # mogon
    
    def read_forces_rtdb(self,outputfile):
        
        with open(outputfile, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.find('dft:gradient') >= 0:
                gradient=[]
                for j in range(i+1, i+1+self.nlines):
                    word = (lines[j].split())
                    gradient.extend([float(word[k]) for k in range(len(word))]) 
        gradient = np.reshape(gradient,(self.natoms,3))                 
        forces = -gradient * Hartree / Bohr
        return forces
    
    def writeforces(self):

        for dispName, atoms in self.vibobj.iterdisplace(inplace=True):
            filename = dispName + '.pckl'
            if not os.path.exists(filename):
                forces = self.read_forces_rtdb(dispName+'_dft.out')
                pickle.dump(forces, open(filename,'wb') , protocol=4)
    
    def dump_nmode(self):

        self.writeforces()
        val, vec = self.projections()
        
        with open ('NMODE_mw.nmode', 'wb') as f:
            pickle.dump(self.natoms,f,protocol=4) 
            pickle.dump(val,f,protocol=4) 
            pickle.dump(vec,f,protocol=4)        
    
#parser = argparse.ArgumentParser(description="Uses equilibrium XYZ file to calculate the projected Hessian and write vibrational modes to .nmode file")
#parser.add_argument("xyzfile", metavar="Geometry (XYZ) file")
#parser.add_argument("-nm", "--nmode", metavar="Dump vibrational output to .nmode file. This is by default 'yes'. If you want to run finite-displacement calculations for the Hessian, use 'No' or 'n' 'N' 'no'")
#args = parser.parse_args()

#H = Hessian(args.xyzfile,1e-3)    
#H.calcfd()
#H.dump_nmode()

#if not os.path.exists(fname):
#    print(fname)
#    H.calcfd()
#else:
#    H.dump_nmode()



