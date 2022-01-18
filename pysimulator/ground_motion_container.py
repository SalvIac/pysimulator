# -*- coding: utf-8 -*-
# pysimulator
# Copyright (C) 2021-2022 Salvatore Iacoletti
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
"""

import numpy as np


class GmfGroupContainer():
    
    def __init__(self, gmf_catalogs, sites, imts):
        # sites, imts to be used in case they are none in GmfCatalogContainer 
        self.gmf_catalogs = gmf_catalogs
        self.sites = sites
        self.imts = imts
        self.num_catalogs = self.get_num_catalogs()
        self.num_sites = self.get_num_sites()
        self.num_imts = self.get_num_imts()
        
    def get_num_catalogs(self):
        return len(self.gmf_catalogs)

    def get_num_sites(self):
        return len(self.sites.array) 

    def get_num_imts(self):
        return len(self.imts.keys())

    def get_float_imt(self):
        T = list()
        for imt in self.imts:
            if imt == "PGA":
                T.append(0.) # 0.01
            else:
                T.append(float(imt[3:-1]))
        return T
    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{} ".format(self.__class__.__name__) + \
               str(self.get_num_catalogs()) + " catalogs of gmfgets" + ">"
   
    def __iter__(self):
        self.__n = 0
        return self

    def __next__(self):
        if self.__n < self.num_catalogs:
            ob = self[self.__n]
            self.__n += 1
            return ob
        else:
            raise StopIteration
    
    def __getitem__(self, i):
        ob = self.gmf_catalogs[i]
        if isinstance(ob, list):
            for o in ob:
                if o.sites is None:
                    o.sites = self.sites
                if o.imts is None:
                    o.imts = self.imts
        else:
            if ob.sites is None:
                ob.sites = self.sites
            if ob.imts is None:
                ob.imts = self.imts
        return ob
    

    def init_filter_time(self, start, end):
        temp = list()
        for gmfs in self.gmf_catalogs:
            dt = gmfs.get_datetimes()
            inds = np.where( np.logical_and(dt>=start, dt<end) )[0]
            datetimes = [dttm for i, dttm in enumerate(gmfs.datetimes) if i in inds]
            gmf = [gm for i, gm in enumerate(gmfs.gmfs) if i in inds]
            if "mainshock" in gmfs.__dict__.keys():
                mainshock = [gm for i, gm in enumerate(gmfs.mainshock) if i in inds]
                temp.append( GmfCatalogContainer(gmf, datetimes,
                                                 mainshock=mainshock) )
            else:
                temp.append( GmfCatalogContainer(gmf, datetimes) )
        return self.__class__(temp, self.sites, self.imts)


    def init_filter_mainshocks(self):
        temp = list()
        for gmfs in self.gmf_catalogs:
            inds = np.where( gmfs.mainshock )[0]
            datetimes = [dttm for i, dttm in enumerate(gmfs.datetimes) if i in inds]
            gmf = [gm for i, gm in enumerate(gmfs.gmfs) if i in inds]
            mainshock = [gm for i, gm in enumerate(gmfs.mainshock) if i in inds]
            temp.append( GmfCatalogContainer(gmf, datetimes,
                                             mainshock=mainshock) )
        return self.__class__(temp, self.sites, self.imts)


    def init_filter_site(self, sites):
        if isinstance(sites, list):
            sites = list(sites)
        temp = list()
        new_sites = self.sites.filtered(sites)
        for gmfs in self.gmf_catalogs:
            datetimes = gmfs.datetimes
            gmf = [gm[:,sites] for i, gm in enumerate(gmfs.gmfs)]
            if "mainshock" in gmfs.__dict__.keys():
                mainshock = gmfs.mainshock
                temp.append( GmfCatalogContainer(gmf, datetimes,
                                                 mainshock=mainshock) )
            else:
                temp.append( GmfCatalogContainer(gmf, datetimes) )
        return self.__class__(temp, new_sites, self.imts)
    
    
    def init_bootstrap(self, size):
        temp = list(np.random.choice(self.gmf_catalogs, size=size, replace=False))
        return self.__class__(temp, self.sites, self.imts)
    
    


class GmfCatalogContainer():
    
    def __init__(self, gmfs, datetimes, sites=None, imts=None, **kwargs):
        '''
        gmfs is a list of gmf (i x s)
        i is the number of imts and s is the number of sites
        sites and imts can be None if they are contained in GmfGroupContainer
        to reduce memory usage
        '''
        if len(datetimes) != len(gmfs):
            raise Exception("inconsistent number of events in datetimes and gmfs")
        self.datetimes = datetimes
        self.gmfs = gmfs
        self.sites = sites
        self.imts = imts
        if self.sites is not None:
            if len(self.sites.array) != self.get_num_sites():
                raise Exception("inconsistent number of sites in gmfs and sites")
        if self.imts is not None:
            if len(self.imts.keys()) != self.get_num_imts():
                raise Exception("inconsistent number of imts in gmfs and imts")
        self.num_events = self.get_num_events()
        self.num_sites = self.get_num_sites()
        self.num_imts = self.get_num_imts()
        self.__dict__.update(kwargs)
        
        
    def get_num_events(self):
        return len(self.datetimes)

    def get_num_sites(self):
        if len(self.gmfs) == 0:
            return 0
        return self.gmfs[0].shape[1]

    def get_num_imts(self):
        if len(self.gmfs) == 0:
            return 0
        return self.gmfs[0].shape[0]

    def get_float_imt(self):
        T = list()
        for imt in self.imts:
            if imt == "PGA":
                T.append(0.) # 0.01
            else:
                T.append(float(imt[3:-1]))
        return T

    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{} ".format(self.__class__.__name__) + \
               str(self.num_events) + " events, " + \
               str(self.num_sites) + " sites, " + \
               str(self.num_imts) + " imts" + ">"

    def iter_imts(self, gmf, imts=None, sites=None):
        if imts is not None:
            li1 = list(imts.slicedic.keys())
        else:
            li1 = list(self.imts.slicedic.keys())
        if sites is not None:
            li2 = sites.array
        else:
            li2 = self.sites.array
        for i in range(0, gmf.shape[0]):
            yield {"imt": li1[i], "site": li2, "gmf": gmf[i,:]}

    def iter_sites(self, gmf, imts=None, sites=None):
        if imts is not None:
            li1 = list(imts.slicedic.keys())
        else:
            li1 = list(self.imts.slicedic.keys())
        if sites is not None:
            li2 = sites.array
        else:
            li2 = self.sites.array
        for i in range(0, gmf.shape[1]):
            yield {"imt": li1, "site": li2[i], "gmf": gmf[:,i]}
    
    def iter_all(self, gmf, imts=None, sites=None):
        if imts is not None:
            li1 = list(imts.slicedic.keys())
        else:
            li1 = list(self.imts.slicedic.keys())
        if sites is not None:
            li2 = sites.array
        else:
            li2 = self.sites.array
        for i in range(0, gmf.shape[0]):
            for j in range(0, gmf.shape[1]):
                yield {"imt": li1[i], "site": li2[j], "gmf": gmf[i,j]}
   
    def __iter__(self):
        self.__n = 0
        return self

    def __next__(self):
        if self.__n < self.num_events:
            out = self[self.__n]
            self.__n += 1
            return out
        else:
            raise StopIteration

    def __getitem__(self, i):
        return self.gmfs[i]
    
    def get_datetimes(self):
        return np.array(self.datetimes)
        
    def get_bulk_data(self):
        out = np.zeros( (self.num_events, self.num_imts, self.num_sites),
                       dtype=np.float32 )
        for i in range(0, self.num_events):
            out[i,:,:] = self.gmfs[i]
        return out
        # out = list() # np.zeros( (self.num_sites, self.num_imts, self.num_events) )
        # for i in range(0, self.num_events):
        #     out.append(self.gmfs[i])
        # return np.array(out)

    def get_site_gmfs(self, s):
        out = list() # np.zeros( (self.num_sites, self.num_imts, self.num_events) )
        for i in range(0, self.num_events):
            out.append(self.gmfs[i][:,s])
        return np.array(out)

