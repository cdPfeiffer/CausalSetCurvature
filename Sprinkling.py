#This file is part of the causal set generator at https://github.com/cdPfeiffer/CausalSetCurvature
#and contains the main classes and methods
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
import networkx as nx
from Causetmethods import *
from methods import *
import functools
import warnings



class Sprinkling():
    """
    Parent class that contains methods that are the same for all Manifolds
    """
    
    def __init__(self,rho,dim=2,depth=1,radius=1):
        assert dim >= 2 and type(dim) is int, "Dimension must be an integer greater or equal 2"
        assert depth>0, "depth has to be bigger than 0"
        self.rho = rho
        self.dim = dim
        self.depth = depth
        self.radius = radius
        
    def draw_t_x(self):
        """
        Plots causet projected on t-x-plane
        """
        ngraph = self.causet.to_networkx()
        nx.draw(ngraph,pos=np.column_stack((self.super_coods[:,2],self.TR[:,0])))
    
    def draw_t_r(self):
        """
        Plots t-r-Plot for causet
        """
        ngraph = self.causet.to_networkx()
        nx.draw(ngraph,pos=np.column_stack((self.TR[:,1],self.TR[:,0])))
    
    def get_eucl_coords(self):
        """
        Return euclidian coordinates in coordinate system used for sprinkling.
        Can be used for further plotting.
        """
        return np.column_stack((self.TR[:,0],self.super_coords[:,2:]))
    
    def sprinkle_T_and_R(self):
        """
        Sprinkles time and radius coord into a Zylinder of depth self.depth and radius self.radius
        Also adds an extra point at t=0,r=0 at the first position of the list
        returns t and r coordinates for sprinkled events
        """
        
        
        #get how many points we need to sprinkle
        N = self.rho*self.depth*Sdminus2(self.dim)/(self.dim-1)*self.radius**(self.dim-1)
        N = np.random.poisson(N)
        
        #distribution coordinates are drawn from should include their influence on the volume, 
        #e.g. r should be drawn from a power distribution with power d-2, angles from a power of sin(theta)
        ts = np.random.uniform(low=-self.depth,high=0,size=N-1)
        ts = np.append(0,ts)
        rs = np.random.power(self.dim-1,size=N-1)*self.radius#this will draw from r^(d-2)
        rs = np.append(0,rs)
        
        return np.column_stack((ts,rs))
        
    def sprinkle_Angles(self):
        """
        Sprinkles anglular coordinates, self.N has to be set
        returns angles for sprinkled events
        """
        #distribution coordinates are drawn from should include their influence on the volume, 
        #e.g. r should be drawn from a power distribution with power d-2, angles from a power of sin(theta)
        angles = []
        if self.dim > 2:
            phis = np.random.uniform(low=0,high=2*np.pi,size=self.N)
            
            if self.dim == 3:
                angles = phis.reshape(-1,1)
            elif self.dim == 4:#use fast way to determine theta
                thetas = randomSinDist(size=self.N)
                angles = np.column_stack((thetas,phis))
                
            else:
                angles = phis
                for i in range(self.dim-3):
                    thetas = randomSinPowNDist(i+1,size=self.N)
                    angles = np.column_stack((thetas,angles))
                    
        return angles
                    
    
    def getRelations(self):
        """
        Puts connections on sprinkled points and does transitive reduction.
        The subclass therefore needs to implement a futureTimelike method
        Return igraph-Graph
        """
        edges = []
        for i in range(self.N):
            for j in range(i+1,self.N):
                causality = self.futureTimelike(i,j)
                if causality == 1:
                    edges.append([i,j])
                elif causality == -1:
                    edges.append([j,i])

        sprinklingGraph = ig.Graph(n=self.N,edges=edges,directed=True)
        return ig_transitive_reduction(sprinklingGraph)
    
    
    @functools.lru_cache(maxsize=1024)
    def preds(self,x):
        """
        returns all predecessors of node x
        """
        return set([n.index for n in self.causet.dfsiter(x,mode="in")])
    
    @functools.lru_cache(maxsize=1024)
    def succs(self,x):
        """
        returns all successors of node x
        """
        return set([n.index for n in self.causet.dfsiter(x,mode="out")])
    
    @functools.lru_cache(maxsize=1024)#esp. usefull if I want to calculate multiple things in simulation
    def get_layers(self,x,layernbrs):
        """
        Returns the layers specified in layernbrs from point nr x in sprinkling
        x: point in sprinkling
        layernbrs: list of layernumbers -- these should be positive numbers
        returns: dict with layernumbers as key and nodes in this layer as value
        """
        assert all(nbr >= 0 for nbr in layernbrs), "Layernumbers have to be non-negative"
        layernbrs = set(layernbrs) #removes duplicates
        layers = {}
        if 0 in layernbrs:
            layers.update({0:set([x])})
            layernbrs.remove(0)
        if 1 in layernbrs:#layer 1 = nodes that are n links away
            layers.update({1:set(self.causet.predecessors(x))})
            layernbrs.remove(1)
        if len(layernbrs) > 0:#if there are other layers then 0 or 1
            for layernbr in layernbrs:
                layers.update({layernbr:set()})
            # get all predecessors (up to rel. level) of x: CS-event in Layer n is maximally n links away
            # therefore use breadth first search and then throw higher layers away
            preds = set()
            for node,distance,p in self.causet.bfsiter(x,mode="in",advanced=True):
                if distance > max(layernbrs):
                    break
                else:
                    preds.add(node.index)
            #loop through them and check if they are part of layer
            for nodeid in preds:
                succ = self.succs(nodeid)
                nodelayer = len(preds.intersection(succ))-1
                if nodelayer in layernbrs:
                    layers[nodelayer].add(nodeid)
        return layers
    
    @functools.lru_cache(maxsize=1024)#esp. usefull if I want to calculate multiple things in simulation
    def layerOf(self,x,y):
        """
        Checks in which layer y is as seen from x
        y has to be in the past of x
        return the layer nbr
        """
        if x == y:
            return 0
        elif y in self.causet.predecessors(x):
            return 1
        else:
            predsOfx = self.preds(x)
            succOfy = self.succs(y)
            return len(predsOfx.intersection(succOfy))-1
        
    @functools.lru_cache(maxsize=1024)#for B_of_B calculation might be useful    
    def B(self,x=0,phi=lambda s,y: -2,epsilon=1):
        """
        Calculated the B operator for the causal set.
        x: starting point, default: 0
        phi: scaler function on the causal set -- a function taking the Sprinkling object and a node-number as arguments; default: phi=-2
        epsilon: smearing factor; default:1
        """
        
        s = 0
        cs = cis(self.dim)
        if epsilon == 1:
            layers = self.get_layers(x,range(1,len(cs)+1))
            for i in range(len(cs)):
                layer = layers[i+1]
                for node in layer:
                    s = s + cs[i]*phi(self,node)
        else:
            for node in self.preds(x):
                s = s + smearingfunction(cs,self.layerOf(x,node)-1,epsilon)*phi(self,node)
        return (epsilon*self.rho)**(2/self.dim)* (alpha(self.dim)*phi(self,x) + epsilon* beta(self.dim)*s)
    
    def B_of_B(self,x=0,phi=lambda s,y: 4,epsilon_outer=1,epsilon_inner=1,epsilon=None):
        """
        Calculated the B(B(phi)) operator for the causal set.
        x: starting point; default: 0
        phi: scaler function on the causal set -- a function taking the Sprinkling object and a node-number as arguments; default: phi=4
        epsilon_outer: set's smearing factor for the outer B-operator; default:1
        epsilon_inner: set's smearing factor for the inner B-operator; default:1
        epsilon: allows to set both smearing factors at the same time
        """
        
        if epsilon is not None:
            epsilon_outer = epsilon
            epsilon_inner = epsilon
    
        s = 0
        cs = cis(self.dim)
        if epsilon_outer == 1:
            layers = self.get_layers(x,range(1,len(cs)+1))
            for i in range(len(cs)):
                layer = layers[i+1]
                for node in layer:
                    s = s + cs[i]*self.B(x=node,phi=phi,epsilon=epsilon_inner)
        else:
            for node in self.preds(x):
                s = s + smearingfunction(cs,self.layerOf(x,node)-1,epsilon_outer)*self.B(x=node,phi=phi,epsilon=epsilon_inner)
        return (epsilon_outer*self.rho)**(2/self.dim)*(alpha(self.dim)*self.B(x=x,phi=phi,epsilon=epsilon_inner) + epsilon_outer* beta(self.dim)*s)
    
    def B_squared(self,x=0,phi=lambda s,y:-2, epsilon1=1,epsilon2=1,epsilon=None):
        """
        Calculated the B(B(phi)) operator for the causal set.
        For this it uses the formula with a sum over sums, so one can compare thisto using the B-operator twice.
        x: starting point; default: 0
        phi: scaler function on the causal set -- a function taking the Sprinkling object and a node-number as arguments; default: phi=4
        epsilon1: set's smearing factor for the first B-operator; default:1
        epsilon2: set's smearing factor for the second B-operator; default:1
        epsilon: allows to set both smearing factors at the same time
        """
        if epsilon is not None:
            epsilon1 = epsilon
            epsilon2 = epsilon
        if epsilon2 == 1 and epsilon1 != 1:
            epsilon2 = epsilon1
            epsilon1 = 1
        
        s1_1=0
        s1_2=0
        s2=0
        cs = cis(self.dim)
        layers = self.get_layers(x,range(1,len(cs)+1)) 
        if epsilon1 == 1:
            for i in range(len(cs)):
                layer = layers[i+1]
                for node in layer:
                    s1_1 = s1_1 + cs[i]*phi(self,node)
                    
            if epsilon2 == 1:
                s1_2=s1_1
                for i in range(len(cs)):
                    for j in range(len(cs)):
                        for node1 in layers[i+1]:
                            for node2 in layers[j+1]:
                                s2 = s2 + cs[i]*cs[j]*phi(self,node1)*phi(self,node2)
            else:
                for node in self.preds(x):
                    s1_2 = s1_2 + smearingfunction(cs,self.layerOf(x,node)-1,epsilon2)*phi(self,node)
                for i in range(len(cs)):
                    for node in layers[i+1]:
                        for node2 in self.preds(x):
                            s2 = s2 + cs[i]*smearingfunction(cs,self.layerOf(x,node2)-1,epsilon2)*phi(self,node2)*phi(self,node)
        else:
            for node in self.preds(x):
                s1_1 = s1_1 + smearingfunction(cs,self.layerOf(x,node)-1,epsilon1)*phi(self,node)
            for node in self.preds(x):
                s1_2 = s1_2 + smearingfunction(cs,self.layerOf(x,node)-1,epsilon2)*phi(self,node)
                
            for node1 in self.preds(x):
                for node2 in self.preds(x):
                    s2 = s2 + smearingfunction(cs,self.layerOf(x,node1)-1,epsilon1)*smearingfunction(cs,self.layerOf(x,node2)-1,epsilon2)\
                                *phi(self,node1)*phi(self,node2)
        
        d=self.dim
        a = alpha(self.dim)
        b = beta(self.dim)
        p = phi(self,x)
        return (epsilon1*epsilon2)**(2/d)*self.rho**(4/d)*(a**2 * p**2 + a*b*p* (epsilon1*s1_1 + epsilon2*s1_2) + b**2 * epsilon1 * epsilon2 * s2)
                    
                
    
    
class MinkowskiSprinkling(Sprinkling):
    
    def __init__(self,rho=1000,dim=2,depth=1,radius=None,form="cone"):
        """
        creates a Sprinkling in Minkowski space
        rho: sprinkling density, default=1000
        dim: dimension of spacetime, default=2, must be integer bigger than 2
        form: "cone", "diamond" or "cylinder"
        depth: depth (eigen time at r=0) from "top" to "bottom" of the form
        radius: max. radius, can be set or is calculated automatically as the max. radius needed
        """
        
        #check inputs
        assert form == "cone" or form == "diamond" or form == "cylinder", "form has to be cone or diamond or cylinder"
        
        #set radius
        if radius is None:
            if form == "diamond":
                radius = depth/2
            else:
                radius = depth  
        
        #init 
        super().__init__(rho,dim=dim,depth=depth,radius=radius)
        
        #create causet
        self.sprinkle(form=form)#sets self.TR, self.angles, self.super_coords
        self.causet = self.getRelations()
        
    def sprinkle(self,form):
        """
        1. sprinkle r and t into a cylinder
        2. cut out form as specified in form
        3. sprinkle the angular coords to the points
        4. convert to Euclidian
        """
        #1
        self.TR = self.sprinkle_T_and_R()
        #2
        if form != "cylinder":#if cone or diamond
            self.TR= self.TR[-self.TR[:,0] >= self.TR[:,1]]#abs(t)>=r
            if form == "diamond":
                self.TR = self.TR[-self.TR[:,0]+self.TR[:,1] <= self.depth]
        self.N = len(self.TR)
        #3
        self.angles = self.sprinkle_Angles()
        #4
        self.super_coords = np.column_stack((self.TR[:,0],np.zeros(self.N),PolarToEuclidian(self.TR[:,1],self.angles)))#add column of zeros to get more compatible with dS and AdS case
        
    def futureTimelike(self,i,j):
        """
        Calculates if coord1 and coord2 are causally related or not and if coord 2 is in the past or the future of 1
        returns 1 if causal and coord 2 in future of coord 1
        returns -1 if causal and coord 2 in past of coord 1
        returns 0 if not causal
        """
        coord1 = self.super_coords[i]
        coord2 = self.super_coords[j]
        ds2 = -(coord1[0]-coord2[0])**2
        ds2 = ds2 + sum((coord1[2:]-coord2[2:])**2)
        if ds2 <= 0:
            if coord2[0] > coord1[0]:
                return 1 #causal and in the future
            else:
                return -1 #causal and in the past
        else:
            return 0 #spacelike

class DeSitterSprinkling(Sprinkling):
    
    def __init__(self,rho=1000,N=None,dim=2,depth=1,radius=None,l=1,form="cone"):
        """
        creates a Sprinkling in de Sitter space
        rho: sprinkling density, default=1000
        dim: dimension of spacetime, default=2, must be integer bigger than 2
        form: "cone", "diamond" or "cylinder"
        depth: depth (eigen time at r=0) from "top" to "bottom" of the form
        radius: max. radius, can be set or is calculated automatically as the max. radius needed
        l: curvature radius of spacetime, should be positive, otherwise absolute value is taken
        """
        #check inputs
        assert form == "cone" or form == "diamond" or form == "cylinder", "form has to be cone or diamond or cylinder"
        self.l = np.abs(l)
        
        #set radius
        if radius is None:
            tau = depth
            if form == "diamond":
                tau = depth/2
            radius = l*np.tanh(tau/l)
                                                           
        #init 
        super().__init__(rho,dim=dim,depth=depth,radius=radius)
        
        #create causet
        self.sprinkle(form=form)#sets self.TR, self.angles, self.eucl_coords
        self.causet = self.getRelations()
        
    def sprinkle(self,form):
        """
        1. sprinkle r and t into a cylinder
        2. cut out form as specified in form
        3. sprinkle the angular coords to the points
        4. convert to Euclidian
        """
        #1
        self.TR = self.sprinkle_T_and_R()
        #2
        if form != "cylinder":#if cone or diamond
            self.TR= self.TR[self.TR[:,1] <= self.l*np.tanh(-self.TR[:,0]/self.l)]
            if form == "diamond":
                self.TR = self.TR[self.TR[:,1] <= self.l*np.tanh((self.depth+self.TR[:,0])/self.l)]
        self.N = len(self.TR)
        #3
        self.angles = self.sprinkle_Angles()
        #4
        x = np.sqrt(self.l**2-self.TR[:,1]**2)*np.sinh(self.TR[:,0]/self.l)
        y = np.sqrt(self.l**2-self.TR[:,1]**2)*np.cosh(self.TR[:,0]/self.l)
        z = PolarToEuclidian(self.TR[:,1],self.angles)
        self.super_coords = np.column_stack((x,y,z))
        
    def futureTimelike(self,i, j):
        """
        Calculates if coord1 and coord2 are causally related or not and if coord 2 is in the past or the future of 1
        returns 1 if causal and coord 2 in future of coord 1
        returns -1 if causal and coord 2 in past of coord 1
        returns 0 if not causal
        """
        Z= - self.super_coords[i,0]*self.super_coords[j,0]
        Z = Z + np.inner(self.super_coords[i,1:],self.super_coords[j,1:])
        Z=Z/self.l**2
                                                          
        if Z>=1:
            if self.TR[j,0]>=self.TR[i,0]:
                return 1 #causal and in the future
            else:
                return -1 #causal and in the past
        else:
            return 0 #spacelike
         

class AntiDeSitterSprinkling(Sprinkling):
    
    def __init__(self,rho=1000,N=None,dim=2,depth=1,radius=None,l=1,form="cone"):
        """
        creates a Sprinkling in Anti de Sitter space
        rho: sprinkling density, default=1000
        dim: dimension of spacetime, default=2, must be integer bigger than 2
        form: "cone", "diamond" or "cylinder"
        depth: depth (eigen time at r=0) from "top" to "bottom" of the form
        radius: max. radius, can be set or is calculated automatically as the max. radius needed
        l: curvature radius of spacetime, should be positive, otherwise absolute value is taken
        """
        #check inputs
        assert form == "cone" or form == "diamond" or form == "cylinder", "form has to be cone or diamond or cylinder"
        if form == "diamond":
            if depth > np.pi*l:
                warnings.warn("at this depth the spacetime bountry is already in the causal diamond")
        else:
            if depth > np.pi/2*l:
                warnings.warn("at this depth you already reach the bountry of spacetime and the standard radius becomes infinite")
        self.l = np.abs(l)
        
        #set radius
        if radius is None:
            tau = depth
            if form == "diamond":
                tau = depth/2
            if tau > np.pi/2*l:
                raise("infinite radius not supportet, you might want to rethink your choice of depth or set a custom radius")
            else:
                radius = l*np.tan(tau/l)
                
        #init 
        super().__init__(rho,dim=dim,depth=depth,radius=radius)
        
        #create causet
        self.sprinkle(form=form)#sets self.TR, self.angles, self.eucl_coords
        self.causet = self.getRelations()
        
    def sprinkle(self,form):
        """
        1. sprinkle r and t into a cylinder
        2. cut out form as specified in form
        3. sprinkle the angular coords to the points
        4. convert to Euclidian
        """
        #1
        self.TR = self.sprinkle_T_and_R()
        #2
        #to be done
        if form != "cylinder":#if cone or diamond
            self.TR= self.TR[self.TR[:,1] <= self.l*np.tan(np.minimum(-self.TR[:,0],np.pi/2)/self.l)]#note that tan(pi/2)!=inf due to numerics
            if form == "diamond":
                self.TR = self.TR[self.TR[:,1] <= self.l*np.tan(np.minimum(self.depth+self.TR[:,0],np.pi/2)/self.l)]
        self.N = len(self.TR)
        #3
        self.angles = self.sprinkle_Angles()
        #4
        x = np.sqrt(self.l**2+self.TR[:,1]**2)*np.sin(self.TR[:,0]/self.l)
        y = np.sqrt(self.l**2+self.TR[:,1]**2)*np.cos(self.TR[:,0]/self.l)
        z = PolarToEuclidian(self.TR[:,1],self.angles)
        self.super_coords = np.column_stack((x,y,z))
        
    def futureTimelike(self,i, j):
        """
        Calculates if coord1 and coord2 are causally related or not and if coord 2 is in the past or the future of 1
        returns 1 if causal and coord 2 in future of coord 1
        returns -1 if causal and coord 2 in past of coord 1
        returns 0 if not causal
        """
        Z= - np.inner(self.super_coords[i,:2],self.super_coords[j,:2])
        Z = Z + np.inner(self.super_coords[i,2:],self.super_coords[j,2:])
        Z=-Z/self.l**2
        if Z<=1:
            if self.TR[j,0]>=self.TR[i,0]:
                return 1
            else:
                return -1
        else:
            return 0
