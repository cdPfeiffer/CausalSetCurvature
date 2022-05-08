#This file is part of the causal set generator at https://github.com/cdPfeiffer/CausalSetCurvature
#and contains different functions used in the main program
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

import numpy as np
import igraph as ig

def ig_transitive_reduction(graph):
    """
    The Algorithm for transitive reduction like it is also used from the package networkx adapted for the package igraph
    """
    TR = ig.Graph(n=graph.vcount(),directed=True)
    descendants = {}
    for u in range(graph.vcount()):
        u_nbrs = set(graph.neighbors(u,mode='out'))
        for v in u_nbrs.copy():
            if v in u_nbrs:
                if v not in descendants:
                    descendants[v] = set([n.index for n in graph.dfsiter(v)])-{v}
                u_nbrs -= descendants[v]
        TR.add_edges([(u,v) for v in u_nbrs])
    return TR

def randomSinDist(size = 1):
    """
    returns random numbers in the interval 0 to pi from a sin distribution
    """
    def inversecdf(x):
        #inverse of 1/2(1-cos(x))
        return np.arccos(1-2*x)
    randnbrs = np.random.uniform(size=size)
    return inversecdf(randnbrs)

def randomSinPowNDist(power,size=1):
    """
    returns random numbers in the interval 0 to pi from a sin^power distribution
    uses monte carlo sampling
    """
    def pdf(x):
        return sin(x)**power
    
    results = []
    while n<size:
        x = np.random.uniform(0,np.pi)
        y = np.random.uniform()
        if y<pdf(x):
            results.append(xrndm)
            n = n+1
    return results

def PolarToEuclidian(rs,angles=[]):
    """
    transforms polar coordinates (r,angles) -- where the last angle is from 0 to 2pi -- to euclidian coordinates x_1,...,x_(#angles+1)
    """
    angles = np.transpose(angles)
    if len(angles)==0:#aka if empty
        return rs * np.random.choice([-1,1],size=len(rs))
    
    else:
        d = len(angles)+1
        #x coordinate has sin in all angles
        for angle in angles:
            coords = rs * np.sin(angle)
        #other coordinates have a cos in the last angle and pot. less angles
        for i in range(d-1):
            xs = rs
            for j in range(d-2-i):#number of angles to include with sin
                xs = xs*np.sin(angles[j])
            xs = xs * np.cos(angles[d-2-i])
            coords = np.column_stack((coords,xs))
    return coords

def calculate_rho_from_N(N,form,dim,depth):
    """
    Calculated the density for a Causal set for a target number of N in Minkowski-space
    """
    
    if form == "cone":
        rho = N/(depth**dim * Sdminus2(dim)/(dim*(dim-1)))
    elif form == "diamond":
        rho = N/(2 * (depth/2)**dim * Sdminus2(dim)/(dim*(dim-1)))
    else:
        rho = N/(depth * Sdminus2(dim)/(dim-1))
    return rho
