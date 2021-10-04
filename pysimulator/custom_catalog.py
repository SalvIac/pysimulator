# -*- coding: utf-8 -*-
"""
@author: Salvatore
"""

import numpy as np
import pandas as pd
from openquake.hazardlib.geo.mesh import RectangularMesh
from pyrisk.etas.etas8p.catalog import CatalogueEtas
from pyrisk.etas.utils import longlat2xy


class CustomCatalog():
    
    def __init__(self, date_times, ruptures, **kwargs):
        '''
        all inputs are iterables (e.g., lists, numpy arrays)
        ruptures: iterable of OQ ruptures
        '''
        self.catalog = {"datetime": date_times,
                        "rupture": ruptures}
        for key, value in kwargs.items():
            self.catalog[key] = value
        self.num_events = self.get_num_events()


    @classmethod
    def mesh2geom(cls, mesh):
        if isinstance(mesh, RectangularMesh):
            # print("rupture mesh must be an instance of openquake.hazardlib.geo.mesh.RectangularMesh")
            xs = mesh.lons
            ys = mesh.lats
            zs = mesh.depths
            x_par = xs[1:,1:] - np.diff(xs, axis=0)[:,:-1]/2 - np.diff(xs, axis=1)[:-1,:]/2
            y_par = ys[1:,1:] - np.diff(ys, axis=0)[:,:-1]/2 - np.diff(ys, axis=1)[:-1,:]/2
            dep = zs[1:,1:] - np.diff(zs, axis=0)[:,:-1]/2 - np.diff(zs, axis=1)[:-1,:]/2
            x_par = x_par.flatten()
            y_par = y_par.flatten()
            dep = dep.flatten()
            # # equivalent but slower method
            # x_par = list()
            # y_par = list()
            # dep = list()
            # for ir in range(0, xs.shape[1]-1):
            #     for ic in range(0, xs.shape[0]-1):
            #         x_par.append(np.mean(xs[ic:ic+2, ir:ir+2]))
            #         y_par.append(np.mean(ys[ic:ic+2, ir:ir+2]))
            #         dep.append(np.mean(zs[ic:ic+2, ir:ir+2]))
            # # check
            # from pyrisk.utils.plot_rup_simple import PlotRup
            # pr = PlotRup()
            # pr.plot_mesh(mesh, s=1)
            # pr.plot_xyz(x_par, y_par, dep, s=2)
        else:
            x_par = mesh.lons.flatten()
            y_par = mesh.lats.flatten()
            dep = mesh.depths.flatten()
        return {"lons": x_par.tolist(), "lats": y_par.tolist(), "depths": dep.tolist()}



    @classmethod
    def rup2geom(cls, rupture):
        return cls.mesh2geom(rupture.surface.mesh)
    
    
    @classmethod
    def surf2geom(cls, surface):
        return cls.mesh2geom(surface.mesh)


    @classmethod
    def multisurf2geom(cls, surfaces):
        geometry = dict(lons=list(), lats=list(), depths=list())
        for surf in surfaces:
            temp = cls.surf2geom(surf)
            geometry["lons"].extend( temp["lons"] )
            geometry["lats"].extend( temp["lats"] )
            geometry["depths"].extend( temp["depths"] )
        return geometry
        


    def get_info_from_rups(self):
        if len(self.catalog['datetime']) == 0:
            self.catalog["magnitude"] = np.array([])
            self.catalog["longitude"] = np.array([])
            self.catalog["latitude"] = np.array([])
        else:
            mags = list()
            lons = list()
            lats = list()
            for i, rupture in enumerate(self.catalog['rupture']):
                mags.append(rupture.mag)
                lons.append(rupture.hypocenter.longitude)
                lats.append(rupture.hypocenter.latitude)
            self.catalog["magnitude"] = np.array(mags)
            self.catalog["longitude"] = np.array(lons)
            self.catalog["latitude"] = np.array(lats)



    def get_more_info_from_rups(self):
        if len(self.catalog['datetime']) == 0:
            self.catalog["depth"] = np.array([])
            self.catalog["rake"] = np.array([])
            self.catalog["strike"] = np.array([])
            self.catalog["dip"] = np.array([])
        else:
            deps = list()
            rakes = list()
            strikes = list()
            dips = list()
            for i, rupture in enumerate(self.catalog['rupture']):
                deps.append(rupture.hypocenter.depth)
                rakes.append(rupture.rake)
                strikes.append(rupture.strike)
                dips.append(rupture.dip)
            self.catalog["depth"] = np.array(deps)
            self.catalog["rake"] = np.array(rakes)
            self.catalog["strike"] = np.array(strikes)
            self.catalog["dip"] = np.array(dips)
            
    
    
    def get_ref_sim_start(self):
        sim_start = pd.Series(self.catalog['datetime']).min()
        return sim_start


    
    def get_ref_centroid(self):
        centroid_lon = np.mean(self.catalog['longitude'])
        centroid_lat = np.mean(self.catalog['latitude'])
        return centroid_lon, centroid_lat

    
    
    def get_etas_rel_info(self, mag_threshold, centroid_lon, centroid_lat,
                          sim_start):
        if len(self.catalog['datetime']) == 0:
            self.catalog['tt'] = np.array([])
            self.catalog['xx'] = np.array([])
            self.catalog['yy'] = np.array([])
            self.catalog['mm'] = np.array([])
            self.catalog['geom'] = list()
            self.catalog['isspontaneous'] = list()
        else:
            # preprocess input catalog (relative time, magnitude and distance)
            # convert fault geometry (where available)
            geom = [None]*len(self.catalog['rupture'])
            # self.catalog['geometry'] = [None]*len(self.catalog['rupture'])
            for i, rupture in enumerate(self.catalog['rupture']):
                geometry = None
                if "Point" in rupture.__class__.__name__:
                    continue
                if rupture.surface.__class__.__name__ == "MultiSurface":
                    geometry = self.multisurf2geom(rupture.surface.surfaces)
                else:
                    geometry = self.rup2geom(rupture)
                # self.catalog['geometry'][i] = geometry
                if geometry is not None:
                    proj = longlat2xy(np.array(geometry["lons"]),
                                      np.array(geometry["lats"]),
                                      centroid_lon, centroid_lat,
                                      dist_unit="degree")
                    geom[i] = {'x': proj['x'],
                               'y': proj['y'],
                               'depth': geometry["depths"]}
            self.catalog['tt'] = CatalogueEtas.date2day(
                                               pd.Series(self.catalog['datetime']),
                                               sim_start).to_numpy()
            new_xy = longlat2xy(self.catalog['longitude'],
                                self.catalog['latitude'],
                                centroid_lon, centroid_lat,
                                dist_unit="degree")
            self.catalog['xx'] = new_xy["x"]
            self.catalog['yy'] = new_xy["y"]
            self.catalog['mm'] = np.array(self.catalog['magnitude'])-mag_threshold
            self.catalog["geom"] = geom
            # isspontaneous -> mainshock
            # par_main_mag -> if not spontaneous, then this is the magnitude of the parent mainshock
            if "mainshock" in self.catalog.keys():
                self.catalog['isspontaneous'] = self.catalog['mainshock']
                self.catalog['par_main_mag'] = [None]*self.catalog['mm'].shape[0] #TODO this should point to the parent
            else: # assumption all events in catalog are mainshocks
                self.catalog['isspontaneous'] = [True]*self.catalog['mm'].shape[0]
                self.catalog['par_main_mag'] = [None]*self.catalog['mm'].shape[0]
            # geometries.append(cls.rup2geom(rupture))
            #     ge = {"lons": list(), "lats": list(), "depths": list()}
            #     for ru in rupture_list[ind].ruptureList[0].ruptureSurface.surfaces:
            #         ge1 = CustomCatalog.mesh2geom(ru.mesh)
            #         ge["lons"].extend(ge1["lons"].tolist())
            #         ge["lats"].extend(ge1["lats"].tolist())
            #         ge["depths"].extend(ge1["depths"].tolist())
            #     ge["lons"] = np.array(ge["lons"])
            #     ge["lats"] = np.array(ge["lats"])
            #     ge["depths"] = np.array(ge["depths"])
            #     geometries.append( ge )



    def process_catalog_4etas(self, mag_threshold, 
                              centroid_lon=None, centroid_lat=None,
                              sim_start=None):
        self.get_info_from_rups()
        if sim_start is None:
            sim_start = self.get_ref_sim_start()
        if (centroid_lon is None) or (centroid_lat is None):
            centroid_lon, centroid_lat = self.get_ref_centroid()
        self.get_etas_rel_info(mag_threshold, centroid_lon, centroid_lat,
                               sim_start)


    
    @classmethod
    def datetime2tdt(cls, sim_start, sim_end, sim_reference):
        tdt = CatalogueEtas.date2day(pd.Series([sim_start, sim_end]),
                                     sim_reference).to_numpy()
        return tdt[0], tdt[1]    
    
    
    def get_num_events(self):
        return len(self.catalog["datetime"])
    
    
    def get_df(self):
        return pd.DataFrame(self.catalog)
    

    def __str__(self):
        string = "<CustomCatalog " + str(self.num_events) + " events>"
        return string
        
        
    def __repr__(self):
        return self.__str__()

    # @attribute
    # def __name__(self):
    #     return "CustomCatalog"
    
    
    def __iter__(self):
        self.__n = 0
        return self

    def __next__(self):
        if self.__n < self.num_events:
            out = {key: value[self.__n] for key, value in self.catalog.items()}
            self.__n += 1
            return out
        else:
            raise StopIteration    
    
    
    
    def simple_plot3d(self):
        from pyrisk.utils.plot3d_rup_simple import PlotRup
        pr = PlotRup()
        for e in range(0, self.get_num_events()):
            pr.plot_xyz(self.catalog["rupture"][e].hypocenter.longitude,
                        self.catalog["rupture"][e].hypocenter.latitude,
                        self.catalog["rupture"][e].hypocenter.depth,
                        s=10, c="r")
            if self.catalog["rupture"][e].__class__.__name__ != "PointRupture":
                if self.catalog["rupture"][e].surface.__class__.__name__ == "MultiSurface":
                    surfaces = self.catalog["rupture"][e].surface.surfaces
                else:
                    surfaces = [self.catalog["rupture"][e].surface]
                for surf in surfaces:
                    pr.plot_mesh(surf.mesh, s=1, c="b")
                # # geometry
                # pr.plot_xyz(cat.catalog["geometry"][e]["lons"],
                #             cat.catalog["geometry"][e]["lats"],
                #             np.array(cat.catalog["geometry"][e]["depths"]),
                #             s=1, c="g")
        pr.show()



    def simple_plot2d(self):
        from pyrisk.utils.plot2d_rup_simple import PlotRup
        pr = PlotRup()
        for e in range(0, self.get_num_events()):
            pr.plot_xyz(self.catalog["rupture"][e].hypocenter.longitude,
                        self.catalog["rupture"][e].hypocenter.latitude,
                        s=10, c="r")
            if self.catalog["rupture"][e].__class__.__name__ != "PointRupture":
                if self.catalog["rupture"][e].surface.__class__.__name__ == "MultiSurface":
                    surfaces = self.catalog["rupture"][e].surface.surfaces
                else:
                    surfaces = [self.catalog["rupture"][e].surface]
                for surf in surfaces:
                    pr.plot_mesh(surf.mesh, s=1, c="b")
                # # geometry
                # pr.plot_xyz(cat.catalog["geometry"][e]["lons"],
                #             cat.catalog["geometry"][e]["lats"],
                #             s=1, c="g")
        pr.show()
        return pr.ax



    def filter(self, bools):
        if isinstance(bools, list):
            bools = np.array(bools)
        if bools.shape[0] != self.num_events:
            raise Exception("boolean vector not the same length of existing catalog")
        inds = np.where(bools)[0]        
        date_times = [dt for d, dt in enumerate(self.catalog["datetime"]) if d in inds]
        ruptures = [ru for r, ru in enumerate(self.catalog["rupture"]) if r in inds]
        other = {}
        for key in self.catalog:
            if key not in ["datetime", "rupture"]:
                other[key] = [ge for r, ge in enumerate(self.catalog[key]) if r in inds]
        return self.__class__(date_times, ruptures, **other)
       
    
    def count_full_ruptures(self):
        return len([rup for rup in self.catalog["rupture"] 
                    if rup.__class__.__name__ != "PointRupture"])
    
