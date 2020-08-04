import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from ase.io import read
import nwchem_nmode_parser as nmode
import nwchem_movecs_parser as nwcparse
import os
import nwchem_zora_parser as zora

nm = nmode.ReadNmode('./0.02/NMODE_mw.nmode')
vibEval = nm.vibEval[6:]

class SpinPhonon():
    def __init__():


    def SOCBasisTransform(mov,zora,PTO=1):
        psia = np.matrix(mov.psia); psib = np.matrix(mov.psib)
        Lx = zora.get("so_x"); Ly = zora.get("so_y"); Lz = zora.get("so_z")
        L_plus = Lx + 1j*Ly; L_minus = Lx - 1j*Ly
        H_ud = np.matmul(psia.getH(),np.matmul(L_minus,psib))
        H_uu = np.matmul(psia.getH(),np.matmul(Lz,psia))
        if PTO==2:
            return np.array([H_uu,H_ud])
        else:
            return (H_ud)
        
    def calc_spinphononCART(d,PTO=2):
        xyz = read('vib.eq.xyz')
        nat = xyz.get_number_of_atoms()
        spinphononcart=[]
        if PTO==2:
            for a in range(nat):
                for i in 'xyz':
                    zora_f = zora.read_nwchem_zora('vib.'+str(a)+str(i)+'+_sodft.zora_so')
                    movecs_f = nwcparse.read('vib.'+str(a)+str(i)+'+_dft.movecs')
                    soc_f = SOCBasisTransform(movecs_f,zora_f,PTO=2)
                    zora_b = zora.read_nwchem_zora('vib.'+str(a)+str(i)+'-_sodft.zora_so')
                    movecs_b = nwcparse.read('vib.'+str(a)+str(i)+'-_dft.movecs')
                    soc_b = SOCBasisTransform(movecs_b,zora_b,PTO=2)
                    sp = 0.5 * (soc_f - soc_b) / d
                    spinphononcart.append(sp)
            spinphononcart = np.array(spinphononcart)
        np.save('SP_cart.npy',spinphononcart)
    
        
    def spcart2norm(spcart):
        xyz = read('vib.eq.xyz')
        m = xyz.get_masses()
        nat = xyz.get_number_of_atoms()
        nm = nmode.ReadNmode('NMODE_mw.nmode')
        amu=sc.physical_constants['atomic mass constant'][0]
        evtoHz = sc.physical_constants['electron volt-hertz relationship'][0]
        HtoeV = sc.physical_constants['Hartree energy in eV'][0]
        vibEval = nm.vibEval
        vibEvec = np.reshape(nm.vibEvec,(nat*3,nat,3))
        nmo = 884
        spnorm = np.zeros((3*nat,2,nmo,nmo),dtype=np.complex128)
        spcart = np.reshape(spcart,(nat,3,2,nmo,nmo))
        
        for l in range(3*nat):
            for a in range(nat):
                for i in range(3):
                    spnorm[l] += spcart[a,i] * vibEvec[l,a,i] * np.sqrt(sc.hbar/(2*m[a]*amu*vibEval[l]*evtoHz)) * 1e10 * HtoeV
            np.save('SP_norm.npy',spnorm)
        return spnorm


disp = [0.02]
path = os.getcwd()+"/"

spnormdisp=[]
for i in disp:
    print(i)
    subdir = path + str(i)
    os.chdir(subdir)
    if not os.path.isfile('SP_cart.npy'):
        calc_spinphononCART(d=i,PTO=2) 
        spnorm=spcart2norm(np.load('SP_cart.npy'))
    elif not os.path.isfile('SP_norm.npy'):
        spnorm = spcart2norm(np.load('SP_cart.npy'))
    else:
        spnorm = np.load('SP_norm.npy')
    spnormdisp.append(spnorm[6:])


spnormdisp = np.array(spnormdisp)
np.save('spnorm_disp.npy',spnormdisp)


spnormdisp = np.load('spnorm_disp.npy')

disp=[0.02]
spph = []
for i in range(np.size(disp)):
    dictspph = {}
    for j in l:
        dictspph[j] = spnormdisp[i,j]
    spph.append(dictspph)

def calc_t1(disp,delta): # delta in cminv 
    broad = delta * cminToeV
    vibInd=[]; 
    Ez = [0.3 * 2 * muB]
#     for m in Ez:
#         vibInd.extend( [[ind, jind] for ind, i in enumerate(vibEval) for jind, j in enumerate(vibEval) if abs(abs(j-i)-m) < broad and i!=j ] )
    vibInd = np.array([[ind,jind] for ind,i in enumerate(vibEval) for jind, j in enumerate(vibEval) if jind > ind])
#     del vibInd[1::2]
    vibInd = np.array(vibInd)
    l = np.array(tuple(set(vibInd.flatten())))
    spph = []
    for i in range(np.size(disp)):
        dictspph = {}
        for j in l:
            dictspph[j] = spnormdisp[i,j]
        spph.append(dictspph)
        t1_disp=[]
    # for j in range(10):
    for i in range(np.size(disp)):
        dictSpPh = spph[i]
        t1=[]
        for t in temp:
            rate_ae = 0; rate_ea = 0
            for i in vibInd:
                M = np.matmul(dictSpPh[i[0]][0,idxHomo,idxHomo+1:]/elEvalDiff[idxHomo+1:], dictSpPh[i[1]][1,idxHomo+1:,idxHomo]) + \
                    np.matmul(dictSpPh[i[0]][1,idxHomo,idxHomo+1:]/elEvalDiff[idxHomo+1:], dictSpPh[i[1]][0,idxHomo+1:,idxHomo])
                rate_ae += (2*pi/hbar) * (bose(vibEval[i[0]],t)) * (bose(vibEval[i[1]],t)+1) * gauss(broad,vibEval[i[1]]-vibEval[i[0]] - Ez[0]) * \
                    (M * M.conj()) 
                M = np.matmul(dictSpPh[i[0]][0,idxHomo,:idxHomo]/elEvalDiff[:idxHomo], dictSpPh[i[1]][1,:idxHomo,idxHomo]) + \
                    np.matmul(dictSpPh[i[0]][1,idxHomo,:idxHomo]/elEvalDiff[:idxHomo], dictSpPh[i[1]][0,:idxHomo,idxHomo])
                rate_ea += (2*pi/hbar) * (bose(vibEval[i[0]],t)+1) * (bose(vibEval[i[1]],t)) * gauss(broad,vibEval[i[1]]-vibEval[i[0]] - Ez[0]) * \
                    (M * M.conj()) 
            t1.append((rate_ea+rate_ae)**-1)
        t1_disp.append(t1)
#         t1_disp = np.array(t1_disp)
    return t1_disp


### ISR rates calculate
disp=[0.01]
delr=[0.3,0.5,1,2,5,10,100]
t1_del = []
for i in delr:
    t1_del.append(calc_t1(disp,i))
# t1_del = np.array(t1_del)


####plot
# for i in range(6):
x=[300];y = [1e-7]
plt.title(r'DNSS; B = 0.3 T')
plt.xlabel(r'Temp [K]')
plt.ylabel(r'$T_1$ [s]')
# plt.xlim(140,420)
# plt.ylim(1e-9,1e3)
# exp = np.loadtxt('vopc_exp.txt')
plt.semilogy(x,y,'.',label='Exp')

for ind, i in enumerate(delr):
    plt.semilogy(temp,t1_del[ind][0],'-',label=r'$\delta=$'+str(i)+r' cm$^{-1}$')
    
plt.legend()
plt.savefig('T1_temp_DNSS.pdf')