#This file is part of the causal set generator at https://github.com/cdPfeiffer/CausalSetCurvature
#and contains different functions for the causal set d'Alembertian
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
from scipy.special import gamma, comb, factorial
import functools

@functools.lru_cache(maxsize=16)
def Sdminus2(d):
    """
    Volume/Area of a d-2-Sphere
    """
    return 2*np.pi**((d-1)/2)/gamma((d-1)/2)

@functools.lru_cache(maxsize=16)
def c(d):
    return Sdminus2(d)/(d*(d-1)*2**(d/2-1))

@functools.lru_cache(maxsize=16)
def alpha(d):
    alpha = -c(d)**(2/d)/gamma((d+2)/d)
    if d % 2 == 0:
        alpha = alpha *2
    return alpha

@functools.lru_cache(maxsize=16)
def beta(d):
    if d%2 == 0:
        return 2*gamma(d/2+2)*gamma(d/2+1)/(gamma(2/d)*gamma(d))*c(d)**(2/d)
    else:
        return (d+1)/(2**(d-1)*gamma(2/d+1))*c(d)**(2/d)

@functools.lru_cache(maxsize=16)
def cis(d):
    nd = d//2 + 2
    cis = []
    if d %2 == 0:
        for i in range(nd):
            ci = 0
            for k in range(i+1):
                ci = ci + comb(i,k)*(-1)**k*gamma(d/2*(k+1)+2)/(gamma(d/2+2)*gamma(1+d*k/2))
            cis.append(ci)
    else:
        for i in range(nd):
            ci = 0
            for k in range(i+1):
                ci = ci + comb(i,k)*(-1)**k*gamma(d/2*(k+1)+3/2)/(gamma((d+3)/2)*gamma(1+d*k/2))
            cis.append(ci)
    return tuple(cis)

@functools.lru_cache(maxsize=128)
def smearingfunction(cs,n,eps):
    f = 0
    for i in range(len(cs)):
        if n>=i:#otherwise it is zero, because it is prefac * n*(n-1)*...(n-i+1)
            prod = 1
            for j in range(i):#pot. more numerically stable then n!/(n-1)!
                prod = prod * (n-j)
            f = f + cs[i]/factorial(i)*(eps/(1-eps))**i * prod
    return (1-eps)**n *f
