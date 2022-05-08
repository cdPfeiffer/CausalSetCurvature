#This file is part of https://github.com/cdPfeiffer/CausalSetCurvature
#and contains an example application
#Copyright (C) 2022 Christopher-Dustin Pfeiffer
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from Sprinkling import *
from joblib import Parallel, delayed
import numpy as np
import time
import os.path as path

#set parameters
N_to_add=600 #number af causal sets to simulate (or add to an existing savefile)
folder="./results/" #folder for savefile
filename="Mnk" #name for savefile
#parameters for sprinkling region
dim=2 
rho = 600 
form="diamond"
depth=np.sqrt(2)
#list of parameters for smearing
es = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]
#number of jobs (should be equal to number of cores to run on)
jobs = 4


#target curvature (not relevant for Minkowski)
R=5
l=np.sqrt(dim*(dim-1)/R)

#defing a function for phi(x)
af=4
cf=12
def phif(sprinkling,node):
    t=sprinkling.TR[node][0]
    return af * t**2 + cf

############################################################
#where the work is done
start = time.time()
def generateSprinklings(i):
    if i%100==1:
        eta = (time.time()-start)/i * (N_to_add-i) #eta in sec
        eta = time.localtime(time.time()+eta)
        """#in case one wants to use a logfile instead
        with open("log.txt","a") as logfile:
            logfile.write("after {}; ETA: {}.{} {}:{}\n".format(i,eta.tm_mday,eta.tm_mon,eta.tm_hour,eta.tm_min))
        """
        print("after {}; ETA: {}.{} {}:{}".format(i,eta.tm_mday,eta.tm_mon,eta.tm_hour,eta.tm_min))

    sprinkling = MinkowskiSprinkling(rho=rho,dim=dim,depth=depth,form=form)

    Bs = []
    B2s = []
    Bfs = []
    B2fs = []
    #BBs = []
    #BBfs = []
    for e in es:
        Bs.append(sprinkling.B(epsilon=e**2))
        B2s.append(sprinkling.B_of_B(epsilon=e**2))
        Bfs.append(sprinkling.B(epsilon=e**2,phi=phif))
        B2fs.append(sprinkling.B_of_B(epsilon=e**2,phi=phif))
        #BBs.append(sprinkling.B_squared(epsilon=e**2))
        #BBfs.append(sprinkling.B_squared(epsilon=e**2,phi=phif))
    return Bs,B2s, Bfs, B2fs


with Parallel(n_jobs=jobs) as parallel:
    results = parallel(delayed(generateSprinklings)(i) for i in range(N_to_add))

if path.exists(folder+filename+".npz"):
    with np.load(folder+filename+".npz") as save:
        Bs = save['Bs']
        B2s = save['B2s']
        Bfs = save['Bfs']
        B2fs = save['B2fs']
        #BBs = save['BBs']
        #BBfs = save['BBfs']
    
    results = np.array(results)
    Bs = np.concatenate((Bs,results[:,0]))
    B2s = np.concatenate((B2s,results[:,1]))
    Bfs = np.concatenate((Bfs,results[:,2]))
    B2fs = np.concatenate((B2fs,results[:,3]))
    #BBs = np.concatenate((BBs,results[:,4]))
    #BBfs = np.concatenate((BBfs,results[:,5]))
else:
    results = np.array(results)
    Bs = results[:,0]
    B2s = results[:,1]
    Bfs = results[:,2]
    B2fs = results[:,3]
    #BBs = results[:,4]
    #BBfs = results[:,5]

np.savez(folder+filename, Bs=Bs, B2s=B2s,Bfs=Bfs,B2fs=B2fs)#, BBs=BBs, BBfs=BBfs)

"""
with open("log.txt","a") as logfile:
    logfile.write("!DONE!\n")
    logfile.write(filename)
    logfile.write("\n\n")
"""
print("!DONE!")
print(filename)
print(len(Bs))
