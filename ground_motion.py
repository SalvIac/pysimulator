# -*- coding: utf-8 -*-
"""
@author: Salvatore
"""

from tqdm import tqdm
import numpy as np
from scipy.stats import norm

from openquake.hazardlib.imt import PGA, PGV, SA
from openquake.hazardlib import const
from openquake.hazardlib.contexts import DistancesContext, RuptureContext
from openquake.hazardlib.geo import Mesh


'''
Note that this class was tailored on the Bradley 
'''

class GroundMotion():

    catalog = None
    
    def __init__(self, rupture_list, sites_coords, sites, gmpe, intensity_measure_types,
                 uncertainty=True, seed=None):
        self.sites_coords = sites_coords
        self.mesh_coords = Mesh(np.array(sites_coords[0]),
                                np.array(sites_coords[1]))    #TODO
        self.gmpe = gmpe
        self.intensity_measure_types = intensity_measure_types
        self.sites = sites
        self.uncertainty = uncertainty
        if seed is not None:
            np.random.seed(seed)
        # get distances once (for speed)
        self.get_distances(rupture_list.ruptures)



    def get_distances(self, rupture_list): #TODO
        # get distances (for speed)
        self.distance_dict = dict()
        self.rjb_dict = dict()
        self.rrup_dict = dict()
        self.rx_dict = dict()
        self.ztor_dict = dict()
        self.ave_rake = dict()
        self.ave_dip = dict()
        for s, source in tqdm(enumerate(rupture_list)):
            rjb = source.surface.get_joyner_boore_distance(
                                                            self.mesh_coords)
            rrup = source.surface.get_min_distance(
                                                              self.mesh_coords)
            rx = source.surface.get_rx_distance(
                                                              self.mesh_coords)
            ztor = source.surface.get_top_edge_depth()
            self.rjb_dict[s] = rjb
            self.rrup_dict[s] = rrup
            self.rx_dict[s] = rx
            self.ztor_dict[s] = ztor
            self.ave_rake[s] = source.rake
            self.ave_dip[s] = source.surface.get_dip()
            
            

    def get_ground_motion(self, catalog):
        self.catalog = catalog
        self.catalog.ground_motions = list()
        # get ground motions
        for e, eve in enumerate(self.catalog.ruptures):
            
            # get rup information
            rake = self.ave_rake[e]
            dip = self.ave_dip[e]
            rjb = self.rjb_dict[e]
            rrup = self.rrup_dict[e]
            rx = self.rx_dict[e]
            ztor = self.ztor_dict[e]
            mag = eve.mag
            
            # classes for gmpe
            rup = RuptureContext()
            setattr(rup, 'mag', np.array([mag]))
            setattr(rup, 'rake', np.array([rake]))
            setattr(rup, 'dip', np.array([dip]))
            setattr(rup, 'ztor', np.array([ztor]))
            dists = DistancesContext()
            # setattr(dists, 'rjb', dist)
            setattr(dists, 'rrup', rrup)
            setattr(dists, 'rjb', rjb)
            setattr(dists, 'rx', rx) 
            stddev_types = [getattr(const.StdDev, 'TOTAL')]
            
            self.catalog.ground_motions.append( dict() )
            for key in self.intensity_measure_types:
                # gmpe
                if key == 'PGA':
                    mean, stddevs = self.gmpe.get_mean_and_stddevs(self.sites,
                                                                   rup,
                                                                   dists,
                                                                   PGA(),
                                                                   stddev_types)
                else:
                    period = float(key[key.find('(')+1 : key.find(')')])
                    mean, stddevs = self.gmpe.get_mean_and_stddevs(self.sites,
                                                                   rup,
                                                                   dists,
                                                                   SA(period),
                                                                   stddev_types)
                # ground motion including random epsilon
                if self.uncertainty:
                    eps = norm.rvs() # epsilon
                else:
                    eps = 1. # 0.
                self.catalog.ground_motions[-1][key] = np.exp(mean+eps*stddevs[0]) # g

   
    def set_seed(self, seed):
        np.random.seed(seed)

   
    
   