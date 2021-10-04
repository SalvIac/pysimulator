# -*- coding: utf-8 -*-
"""
@author: Salvatore
"""

from tqdm import tqdm
import numpy as np


class EpCurves():
    
    def __init__(self, levels, exposure, loss_data):
        self.levels = levels
        self.exposure = exposure
        self.compute_ep_curves(loss_data)
        self.num_catalogs = loss_data.shape[0]
        

    @classmethod
    def get_prob_exceed(cls, gms, level):
        '''
        this gets the probability of exceedance of each im type at each site
        given the ground motion catalog and an im level (in the same units) 
        '''
        temp = np.empty((gms.num_imts, gms.num_sites, gms.num_catalogs), dtype=np.bool)
        for i, g1 in enumerate(gms):
            g1.num_events = g1.get_num_events() #TODO (delete eventually)
            g1.num_sites = g1.get_num_sites() #TODO (delete eventually)
            g1.num_imts = g1.get_num_imts() #TODO (delete eventually)
            temp2 = g1.get_bulk_data()
            temp[:,:,i] = np.any(temp2 > level, axis=0)
        exceeds = np.sum(temp, axis=2)
        prob = exceeds / gms.num_catalogs
        # # this is a much slower (but clearer) version of the above code
        # exceeds2 = np.zeros((gms.num_imts, gms.num_sites))
        # for m in tqdm(range(0, gms.num_imts)):
        #     for s in range(0, gms.num_sites):
        #         for i, g1 in enumerate(gms):
        #             for gmf in g1:
        #                 if gmf[m,s] > level:
        #                     exceeds2[m,s] += 1
        #                     break
        return prob
                

    def compute_hazard_curves(self, gms):
        if len(self.sites.array) != gms.num_sites:
            raise Exception("inconsistent number of sites in gmfs and sites")
        if len(self.imts.keys()) != gms.num_imts:
            raise Exception("inconsistent number of imts in gmfs and imts")
        self.hazard_curves = np.zeros( (len(self.levels),
                                        gms.num_imts,
                                        gms.num_sites) )
        for i, level in tqdm(enumerate(self.levels)):
            prob = self.get_prob_exceed(gms, level)
            self.hazard_curves[i,:,:] = prob
        

    def get_hazard_curve_site_imt_str(self, s, imt):
        i = np.where(imt == np.array(list(self.imts.slicedic.keys())))[0]
        print(i)
        out = {"hazard": self.hazard_curves[:, s, i],
               "levels": self.levels,
               "site": self.sites[s],
               "imt": imt}
        return out

        
    def get_hazard_curve_site_imt(self, s, i):
        li = list(self.imts.slicedic.keys())
        out = {"hazard": self.hazard_curves[:, s, i],
               "levels": self.levels,
               "site": self.sites[s],
               "imt": li[i]}
        return out
        
        
    def get_all_hazard_curves_site(self, s):
        out = {"site": self.sites[s],
               "hazard_curves": list()}
        li = list(self.imts.slicedic.keys())
        for i in range(0, len(li)):
            out["hazard_curves"].append(
                {"hazard": self.hazard_curves[:, s, i],
                 "levels": self.levels,
                 "imt": li[i]} )
        return out

    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<HazardCurves " + \
               str(len(self.sites.array)) + " sites, " + \
               str(len(self.imts.slicedic.keys())) + " imts, " + \
               str(len(self.levels)) + " levels, " + \
               str(self.num_catalogs) + " catalogs" + ">"


