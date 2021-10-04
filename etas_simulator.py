# -*- coding: utf-8 -*-
"""
Module :mod:`pyrisk.simulator.etas_smulator` defines
:class:`EtasSimulator`.
"""

import os
import datetime
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import uniform, poisson
from shapely.geometry import Point
from tqdm import tqdm

from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.pmf import PMF

from pyrisk.etas.utils import xy2longlat, longlat2xy, get_region
from pyrisk.etas.etas8p.etas_mle_bkg import EtasMleBkg
from pyrisk.etas.simulation_functions import (inv_cdf_magnitude_trunc,
                                              inv_cdf_time,
                                              inv_cdf_time_trunc,
                                              inv_cdf_space5,
                                              inv_cdf_space5_trunc,
                                              inv_cdf_space3,
                                              compl_vs_time_p16,
                                              compl_vs_time_hs06,
                                              compl_vs_time_general)
from pyrisk.simulator.custom_catalog import CustomCatalog
from pyrisk.simulator.rupture_builder import RuptureBuilder
from myutils.run_multiprocess import run_multiprocess




class EtasSimulator():
    
    # simulation options
    allowed_keys = ['num_realization', 'multiprocessing', 'cores',
                    'only_first_generation', "folder", 'save_mode']

    allowed_filters = ['sim_start', 'sim_end', 'min_mag', 'region']

    """
    ETAS simulator.
    
    :param params: dict containing the necessary parameters for the ETAS model 
                   used ("model" variable). It can contain:
                   alpha, A (productivity)
                   p, c (Omori's), ta (truncation, optional)
                   D, q, gamma (spatial distribution, some optional)
                   b b-value of the G-R mag-freq distr
                   min_mag lower bound magnitude (used in the ETAS formulation and mag-freq distr)
                   max_mag upper bound magnitude (mag-freq distr)
                   c1, c2, c3 (incompleteness model, optional, Helmstetter default)
                   in the future: mag_empirical
    :param input_catalog: instance of :class:`~pyrisk.simulator.custom_catalog.CustomCatalog`
                          class or list of instaces.
    :param model: string, default: "timetrunc:True_magnitudetrunc:True_spatialpdf:5_incompletess:True"
    :param nodal_planes_distr:
    :param depth_distr:
    :param filters:
    :param simul_options:
    """
    def __init__(self, params, input_catalog, model=None, 
                 nodal_planes_distr=[], depth_distr=[], filters={},
                 simul_options={}):

        # etas model
        self.params = params
        self.model = self._model2dict(model)
        
        if isinstance(input_catalog, list):
            # e.g., add aftershocks to 1000 stoch mainshock-only catalogs
            self.mode = "stoch_catalog" 
            self.input_catalog = input_catalog
        else:
            # e.g., from mainshock create 1000 simulations of aftershock sequence
            self.mode = "single_event" 
            self.input_catalog = [input_catalog]
        
        self.nodal_planes_distr = nodal_planes_distr
        self.depth_distr = depth_distr
        self._check_inputs()
        
        # options (set default and override)
        if simul_options is None:
            simul_options = {}
        simul_options.setdefault('num_realization', 1000)
        simul_options.setdefault('multiprocessing', True)
        simul_options.setdefault('cores', 3)
        simul_options.setdefault('only_first_generation', False)
        simul_options.setdefault('background', None)
        simul_options.setdefault('save_mode', False)
        simul_options.setdefault('folder', None)
        self.simul_options = simul_options     
        self.__dict__.update((k, v) for k, v in simul_options.items() 
                              if k in self.allowed_keys)
        if self.simul_options["save_mode"]:
            self.create_output_folder(self.simul_options["folder"])
            
        # stoch_catalog mode only works if num_realization == len(input_catalog)
        # if self.mode == "stoch_catalog":
        #     if self.num_realization != len(self.input_catalog):
        #         # impose
        #         print("num_realization imposed!")
        #         self.num_realization = len(self.input_catalog)
        
            
        # filters on magnitude, region and time
        # (time the only one affecting the simulations, for speed)
        if filters is None:
            filters = {}
        filters.setdefault('sim_start', None)
        filters.setdefault('sim_end', None)
        filters.setdefault('t_start', None)
        filters.setdefault('t_end', None)
        filters.setdefault('min_mag', None)
        filters.setdefault('region', None)
        self.filters = filters
        
        # placeholders
        self.inputs = [None]*(self.num_realization*len(self.input_catalog))
        self.outputs = [None]*(self.num_realization*len(self.input_catalog))
        self.seed = 1
        
    
    
    def _check_inputs(self):
        # if ((self.sim_start is None) and (self.sim_end is not None)) or \
        #    ((self.sim_start is not None) and (self.sim_end is None)):
        #     raise Exception('check sim_start and sim_end!')
        self._check_params_vs_model()
        # if not isinstance(self.input_catalog, CustomCatalog): # for some reason it doesn't work
        #     raise Exception("input_catalog must be an instance of CustomCatalog")
        for i, _ in enumerate(self.input_catalog):
            if self.input_catalog[i].__class__.__name__ != "CustomCatalog":
                raise Exception("some input_catalog items are not instances of CustomCatalog (mode {})".format(self.mode))
            

    @classmethod
    def _all_keys_included(cls, check, keys):
        for key in check:
            if key not in keys:
                return False
        return True



    def _check_params_vs_model(self):
        keys = self.params.keys()
        mod = self.model
        print(keys)
        if not self._all_keys_included(["alpha", "A", "p", "c", "b", "min_mag"], keys):
            raise Exception("check that all the following variables are included: alpha, A, p, c, b, min_mag")
        if mod["timetrunc"] and ("ta" not in keys):
            raise Exception("for truncated time model, truncation time ta is needed!")
        if not mod["magnitudetrunc"]:
            raise Exception("untruncated magnitude model not yet implemented!")
        if mod["magnitudetrunc"] and ("max_mag" not in keys):
            raise Exception("for the truncated mag-freq distribution max_mag is needed!")
        if mod["spatialpdf"] not in [3,5]:
            raise Exception("model "+str(mod["spatialpdf"])+" for spatial pdf not currently implemented!")
        if (mod["spatialpdf"]==3) and (not self._all_keys_included(["q", "D"], keys)):
            raise Exception("not all inputs needed for model 3 are included (q,D)")
        if (mod["spatialpdf"]==5) and (not self._all_keys_included(["q", "D", "gamma"], keys)):                        
            raise Exception("not all inputs needed for model 5 are included (q,D,gamma)")
        if mod["timetrunc"] and ("ta" not in keys):
            raise Exception("for truncated time model, truncation time ta is needed!")
        if mod["incompletess"] and ("c1" not in keys):             
            self.params["c1"] = 1.
            warnings.warn("default c1 value used for incompleteness model (c1=1)")
        if mod["incompletess"] and ("c2" not in keys):             
            self.params["c2"] = 4.5
            warnings.warn("default c2 value used for incompleteness model (c2=4.5)")
        if mod["incompletess"] and ("c3" not in keys):             
            self.params["c3"] = 0.75
            warnings.warn("default c3 value used for incompleteness model (c3=0.75)")
        if mod["incompletess"] and ("incompl_min_mag" not in keys):             
            self.params["incompl_min_mag"] = 6.
            warnings.warn("default incompl_min_mag value used for incompleteness model (incompl_min_mag=6)")

    




    
    @classmethod
    def _model2dict(cls, model):
        modeldict = {"timetrunc": True,
                     "magnitudetrunc": True,
                     "spatialpdf": 5,
                     "incompletess": True} # default
        # model = "timetrunc:False_magnitudetrunc:False_spatialpdf:3_incompletess:False" # test
        if isinstance(model, str):
            for string in model.split("_"):
                key = string.split(":")[0]
                if key not in modeldict.keys():
                    raise Exception("incorrect key in the model string: "+key)
                strval = string.split(":")[1]
                if strval not in ["True", "False"]: # "True_spatialpdf" is the only int
                    val = int(strval)
                elif strval in ["True"]:
                    val = True
                elif strval in ["False"]:
                    val = False
                modeldict[key] = val
        return modeldict
    
    
    
    
    
    
    def create_output_folder(self, folder):
        if folder is None:
            self.output_folder = os.path.join(os.getcwd())
        else:
            self.output_folder = os.path.join(os.getcwd(), folder)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
    
    
    def slice2inds(self, slice_inputs=None):
        if slice_inputs is None:
            iterator = self.input_catalog
        else:
            iterator = self.input_catalog[slice_inputs]
        inds = list()
        for cat in iterator:
            i = self.input_catalog.index(cat)
            for l in range(0, self.num_realization):
                inds.append((i, l+self.num_realization*i))
        return inds
    
    
    
    def _define_inputs(self):
        # input for simulations
        # self.__dict__.update((k, v) for k, v in self.simul_options.items() 
        #                      if k in self.allowed_keys)
        print("define inputs for etas")
        inds = self.slice2inds()
        for i, ind in tqdm(inds):
            cat = self.input_catalog[i]
            if len(cat.catalog["datetime"]) != 0:
                cat.process_catalog_4etas(self.params["min_mag"])
            _gen0 = cat.get_df()
            
            filters = deepcopy(self.filters)
            if (filters["sim_start"] is not None) and \
               (filters["sim_end"] is not None) and \
               len(cat.catalog['datetime']) != 0:
                tdt = cat.datetime2tdt(
                    filters["sim_start"], filters["sim_end"],
                    pd.Series(cat.catalog['datetime']).min())
                filters["t_start"] = tdt[0]
                filters["t_end"] = tdt[1]
            
            dic = {'gen0': _gen0.copy(), #TODO use dict of numpy arrays (pandas is slow)
                   'params': self.params,
                   'model': self.model,
                   'only_first_generation': self.simul_options['only_first_generation'],
                   'background': self.simul_options['background'],
                   'seed': self.seed*(ind+1),
                   'filters': filters,
                   }
            self.inputs[ind] = dic
        
    

    def simulate(self, slice_inputs=None):
        if self.multiprocessing:
            # define inputs
            if not all(self.inputs):
                self._define_inputs()
            temp = self.slice2inds(slice_inputs)
            sl = slice(np.array(temp)[:,1].min(), np.array(temp)[:,1].max()+1)
            print("simulate etas model")
            out = run_multiprocess(self.simulate_multi, self.inputs[sl],
                                   self.cores)
        else:
            print("simulate etas model")
            out = list()
            inds = self.slice2inds()
            for i, ind in tqdm(inds):
                cat = self.input_catalog[i]
                if len(cat.catalog["datetime"]) != 0:
                    cat.process_catalog_4etas(self.params["min_mag"])
                _gen0 = cat.get_df()
                
                filters = deepcopy(self.filters)
                if (filters["sim_start"] is not None) and \
                    (filters["sim_end"] is not None) and \
                    len(cat.catalog['datetime']) != 0:
                    tdt = cat.datetime2tdt(
                        filters["sim_start"], filters["sim_end"],
                        pd.Series(cat.catalog['datetime']).min())
                    filters["t_start"] = tdt[0]
                    filters["t_end"] = tdt[1]
                
                dic = {'gen0': _gen0.copy(), #TODO use dict of numpy arrays (pandas is slow)
                        'params': self.params,
                        'model': self.model,
                        'only_first_generation': self.simul_options['only_first_generation'],
                        'background': self.simul_options['background'],
                        'seed': self.seed*(ind+1),
                        'filters': filters,
                        }
                self.inputs[ind] = {'seed': self.seed*(ind+1)}
                out.append( self.simulate_multi(dic) )
            # for inpu in tqdm(self.inputs[sl]):
            #     out.append( self.simulate_multi(inpu) )
                
        # self.intermediate = out
        catalogs = self.convert2catalogs(out, self.inputs)
        self.outputs = catalogs
        return catalogs
    
    
    
    def convert2catalogs(self, out, inputs):
        inps = list()
        for j, o in enumerate(out):
            seed = inputs[j]['seed']
            np.random.seed(seed=seed*2)
            depth = self.get_depth(o.shape[0])
            nodpl = self.get_nodalplane(o.shape[0])
            inps.append([o, depth, nodpl])
        print("post-processing")
        if self.multiprocessing:
            catalogs = run_multiprocess(self.get_custom_catalog, inps,
                                        self.cores)
        else:
            catalogs = list()
            for inpu in tqdm(inps):
                catalogs.append( self.get_custom_catalog(inpu) )
        return catalogs

    
    
    @classmethod
    def get_custom_catalog(cls, inputs):
        o = inputs[0]
        depth = inputs[1]
        nodpl = inputs[2]
        date_times = list()
        ruptures = list()
        for i in range(0, o.shape[0]):
            date_times.append(o["datetime"].iloc[i])
            if not o["isspontaneous"].iloc[i] or \
               o["isspontaneous"].iloc[i] == "bkg":
                ruptures.append( RuptureBuilder.init_point(
                                                o["magnitude"].iloc[i],
                                                o["longitude"].iloc[i],
                                                o["latitude"].iloc[i],
                                                depth[i],
                                                nodpl[i].rake,
                                                nodpl[i].strike,
                                                nodpl[i].dip) )
            else:
                ruptures.append( o["rupture"].iloc[i] )
        return CustomCatalog(date_times, ruptures,
                             mainshock=o["isspontaneous"].to_list())
        
    
    
    
    def get_depth(self, num):
        return self.depth_distr.sample(num)

    
    def get_nodalplane(self, num):
        return self.nodal_planes_distr.sample(num)


    def set_seed(self, seed):
        self.seed = seed

        
    # def _lonlat2xy(self, lons, lats):
    #     proj = CatalogueEtas.longlat2xy(long=np.array(lons),
    #                                     lat=np.array(lats),
    #                                     region_poly=self.fit.catalog.region_poly,
    #                                     dist_unit=self.fit.catalog.dist_unit)
    #     coords = [(x, y) for x,y in zip(proj['x'], proj['y'])]
    #     region_win = Polygon(coords)       
    #     return region_win
 
    
        
    def __str__(self):
        string = "<EtasSimulator"+" \n"\
            "mode: "+self.mode+"\n"+\
            "model: "+str(self.model)+"\n"+\
            "parameters: "+str(self.params)+"\n"+\
            "input catalog: "+str(self.input_catalog)+"\n"+\
            "simulation options: "+str(self.simul_options)+">"
        return string
    
    
    def __repr__(self):
        return self.__str__()        


    
    
    # simulate
    @staticmethod
    def simulate_multi(inputs):
        gen0 = inputs['gen0']
        params = inputs['params']
        model = inputs['model']
        mag_threshold = params['min_mag']
        only_first_generation = inputs['only_first_generation']
        background = inputs['background']
        seed = inputs['seed']
        filters = inputs['filters']
        
        # for reproducibility
        np.random.seed(seed=seed)
        
        if background is not None:
            if background["type"] == "gridded":
                bkgr = EtasSimulator.get_background_gridded(background, gen0,
                                                            params, filters)
            elif background["type"] == "zhuang":
                bkgr = EtasSimulator.get_background_zhuang(fit, t, tdt, buffer_region_xy)
            gen1 = pd.concat([gen0, bkgr], ignore_index=True)
        else:
            gen1 = gen0
        
        if gen1.shape[0] == 0:
            return gen1
        gg = [gen1] # list with each generation, regardless the parent event
        
        timedep_mc = None
        if model["incompletess"]:
            # timedep_mc = EtasSimulator.calc_timedep_mc(params, gg[-1], gg[-1],
            #                                            mag_threshold, 5)
            timedep_mc = EtasSimulator.calc_timedep_mc(params,
                                                        gg[-1].index,
                                                        gg[-1]["tt"].values,
                                                        gg[-1]["tt"].values,
                                                        gg[-1]["magnitude"].values,
                                                        mag_threshold)
        
        gl = EtasSimulator.get_following_generation(gg[-1], params, model,
                                                    timedep_mc, filters)
        if len(gl) != 0:
            gg.append(pd.concat(gl, ignore_index=True))
        
        if not only_first_generation:
            l = 1
            while len(gl) != 0:
                if model["incompletess"]:
                    # timedep_mc = EtasSimulator.calc_timedep_mc(params, gg[-1],
                    #                              pd.concat(gg, ignore_index=True),
                    #                              mag_threshold, 0)
                    timedep_mc = EtasSimulator.calc_timedep_mc(params,
                                                                gg[-1].index,
                                                                gg[-1]["tt"].values,
                                                  pd.concat(gg, ignore_index=True)["tt"].values,
                                                  pd.concat(gg, ignore_index=True)["magnitude"].values,
                                                  mag_threshold)
                    
                gl = EtasSimulator.get_following_generation(gg[-1], params, model,
                                                            timedep_mc, filters)
                if len(gl) != 0:
                    gg.append(pd.concat(gl, ignore_index=True))
                    l += 1
    
        stoch_catalog = pd.concat(gg, ignore_index=True)
        stoch_catalog.drop_duplicates(subset=['tt', 'xx', 'yy', 'mm'], inplace=True)
        stoch_catalog.sort_values(by=['tt'], inplace=True)
        
        # filter magnitude
        stoch_catalog = EtasSimulator.filter_events_mag(stoch_catalog,
                                                        filters["min_mag"])
        
        # convert tt (days) in datetime (no need to filter here)
        stoch_catalog["datetime"] = stoch_catalog["datetime"].iloc[0] - \
                                    datetime.timedelta(days=stoch_catalog["tt"].iloc[0]) + \
                                    np.array([datetime.timedelta(days=days) for days in stoch_catalog["tt"]])
        # filters time
        stoch_catalog = EtasSimulator.filter_events_mag(stoch_catalog,
                                                        filters["min_mag"])
        
        # convert x y in lon lat
        proj = xy2longlat(stoch_catalog['xx'],
                          stoch_catalog['yy'],
                          np.mean(gen0['longitude']),
                          np.mean(gen0['latitude']),
                          dist_unit="degree")
        stoch_catalog['longitude'] = proj['long']
        stoch_catalog['latitude'] = proj['lat']    
        
        # filter space
        stoch_catalog = EtasSimulator.filter_events_space(stoch_catalog,
                                                          filters["region"])
        stoch_catalog = EtasSimulator.filter_events_time(stoch_catalog,
                                                         filters["t_start"],
                                                         filters["t_end"])
        stoch_catalog.reset_index(inplace=True, drop=True)
        return stoch_catalog
    
    
        # filter
        # stoch_catalog = EtasSimulator.filter_events(stoch_catalog, t, tdt, simulation_region_xy, mag_threshold)

        # if save_mode: # save catalog
        #     stoch_catalog.to_csv(path+'\stoch_catalog_'+str(it).zfill(5)+'.csv')
        #     return it
        # else:
            
        # # simulation and buffer region
        # self.simulation_region = simulaton_region
        # self.simulation_region_xy = self._lonlat2xy(simulaton_region['lon'],
        #                                             simulaton_region['lat'])
        # if buffer_region is None:
        #     self.buffer_region = simulaton_region
        # else:
        #     self.buffer_region = buffer_region
        # self.buffer_region_xy = self._lonlat2xy(self.buffer_region['lon'],
        #                                         self.buffer_region['lat'])
        # o = EtasSimulator.filter_events(o, t_start, t_end, buffer_region_xy, mag_threshold)
        
        

    @staticmethod
    def get_following_generation(gen, params, model, timedep_mc, filters):
        ''' For each event i , namely (ti, xi, yi, mi), in the catalog G(l), 
        simulate its N(i) offspring, namely, Oi(l) where N(i) is a Poisson random
        variable with a mean of κ(mi), and tk(i) (xk(i), yk(i)) and mk(i) are
        generated from the probability densities g, f and s respectively. 
        Let Oi(l) ← {(tk , xk , yk , mk)} where tk in [t, t + dt].
        
        # for km (productivity): the expectation of the number of children
        # (unit: events) spawned by an event of magnitude m
        A = fit.param['A']
        alpha = fit.param['alpha']
        
        # for the pdf of the length of the time interval between a child and its
        # parent (unit: day−1)
        c = fit.param['c']
        p = fit.param['p']
        
        # for the pdf of the relative locations between the parent and children
        # (unit: deg−2)
        D = fit.param['D']
        q = fit.param['q']
        gamma = fit.param['gamma']
        '''
        mag_threshold = params['min_mag']
        t_start = filters["t_start"]
        t_end = filters["t_end"]
        
        # account for full fault geometry
        mag_parent = list()
        mag4spat = list()
        time_parent = list()
        x_par = list()
        y_par = list()
        points = list()
        timedep_mc_par = list()
        par_main_mag = list()
        wc = WC1994()
        for par, geom in gen['geom'].items():
            if geom is None:
                mag_parent.append(gen['magnitude'][par])
                mag4spat.append(gen['magnitude'][par])
                time_parent.append(gen['tt'][par])
                x_par.append(gen['xx'][par])
                y_par.append(gen['yy'][par])
                points.append(1)
                if timedep_mc is not None:
                    timedep_mc_par.append(timedep_mc[par])
                else:
                    timedep_mc_par.append(None)
                par_main_mag.append(gen["par_main_mag"][par])
            else:
                for i in range(0, geom['x'].shape[0]):
                    mag_parent.append(gen['magnitude'][par])
                    time_parent.append(gen['tt'][par])
                    x_par.append(geom['x'][i])
                    y_par.append(geom['y'][i])
                    points.append(geom['x'].shape[0])
                    if timedep_mc is not None:
                        timedep_mc_par.append(timedep_mc[par])
                    else:
                        timedep_mc_par.append(None)
                    # mag4spat.append(mag_parent[-1])
                    mag4spat.append( np.max([ mag_threshold,
                                       wc.get_median_mag(wc.get_median_area(mag_parent[-1], None)/points[-1], None) ]))
                    par_main_mag.append(gen["par_main_mag"][par])
                    
        # # productivity (corrected by the number of points UCERF3)
        # km = (params["A"] / np.array(points)) * np.exp(params["alpha"] * \
        #                                                (np.array(mag_parent)-mag_threshold))
        if timedep_mc is None:
            km = (params["A"] / np.array(points)) * np.exp(params["alpha"] * \
                                              (np.array(mag_parent)-mag_threshold))
        else:
            km = (params["A"] / np.array(points)) * np.exp(params["alpha"] * \
                                              (np.array(mag_parent)-np.array(timedep_mc_par)))
    
        # km[np.array(mag_parent) <= 5.49] = 0. 
        # this assumes that the mainshock occurs at zero (time_parent)
        # mag_main = model["magnitude"]["mag_main"]
        # ind = np.array(time_parent) > 0.
        # mc = np.clip(mag_main-4.5-0.75*np.log10(np.array(time_parent)[ind]),
        #              mag_threshold, np.array(mag_parent)[ind])
        # km[ind] = (params["A"] / np.array(points)[ind]) * \
        #                 np.exp(params["alpha"] * (np.array(mag_parent)[ind]-mc))
        
        # random number of offspring events
        ni = poisson.rvs(km)
        if isinstance(ni, int):
            ni = np.array([ni])
    
        ol = list() # list of offsprings for each event of the generation l
        for par in range(0, ni.shape[0]):
            if ni[par] > 0:
                o = EtasSimulator.generate_offsprings(params, model, ni[par],
                                                      time_parent[par],
                                                      x_par[par], y_par[par],
                                                      mag_parent[par],
                                                      mag4spat[par],
                                                      par_main_mag[par])
                                        # timedep_mc_par[par])
                o = EtasSimulator.filter_events_time(o, t_start, t_end)
                # o.sort_values(by=['tt'], inplace=True)
                # o.reset_index(inplace=True, drop=True)
                ol.append(o)
    
        return ol
    
    
    
    @staticmethod
    def filter_events_space(o, region=None):
        if region is not None:
            o = o[ np.array([Point(xxx, yyy).within(region) for xxx, yyy 
                             in zip(o['longitude'], o['latitude'])]) ]
        return o
    
    
    @staticmethod
    def filter_events_mag(o, mag_min=None):
        if mag_min is not None:
            o = o[ o['magnitude'] >= mag_min ]
        return o
    
    
    @staticmethod
    def filter_events_time(o, t_start=None, t_end=None):
        if (t_start is not None) and (t_end is not None):
            o = o[ (o['tt'] >= t_start) & (o['tt'] <= t_end) ]
        if (t_start is not None):
            o = o[ (o['tt'] >= t_start) ]
        if (t_end is not None):
            o = o[ (o['tt'] <= t_end) ]
        return o
    
    
    
    @staticmethod
    def generate_offsprings(params, model, num_offspr, time_par, x_par, y_par,
                            mag_parent, mag4spat, par_main_mag=None):
        b = params['b']
        mag_threshold = params['min_mag']
        mag_max = params['max_mag']
        # this tracks the magnitude of the mainshock causing all the event tree
        if par_main_mag is None:
            par_main_mag = mag_parent
        
        # g = (p - 1)/c * np.power(1. + ttt/c, -p)
        u = uniform.rvs(size=num_offspr)
        # inverse cdf of time interval pdf (g)
        # deltat = params["c"]*((1-u)**(1/(1-params["p"]))-1)
        if model["timetrunc"]:
            ta = params["ta"]
            deltat = inv_cdf_time_trunc(u, params["c"], params["p"], ta)
        else:
            deltat = inv_cdf_time(u, params["c"], params["p"])
        t_offspr = time_par + deltat
        
        # only account for t_offspr after 0. (to speed things up)
        #TODO this does not work if history is not provided until the start of the simulation period
        deltat = deltat[t_offspr >= 0.]
        t_offspr = t_offspr[t_offspr >= 0.]
        num_offspr = t_offspr.shape[0]
        
        if num_offspr > 0:
            # b = (1. / gl['mm'].mean())/np.log(10.)
            mmax = min(par_main_mag, mag_max)
            mmin = mag_threshold
            # m_offspr = mmin + (np.log10(1.-u*(1.-10.**(-b*(mmax-mmin)))))/(-b)
            u = uniform.rvs(size=num_offspr)
            if model["magnitudetrunc"]:
                if not model["incompletess"]:
                    m_offspr = inv_cdf_magnitude_trunc(u, b, mmin, mmax)
                elif model["incompletess"]:
                    coeffs = [params["c1"], params["c2"], params["c3"]]
                    if mag_parent >= params["incompl_min_mag"]:
                        # mc = np.clip(compl_vs_time_hs06(mag_parent, deltat), mmin, mag_parent)
                        mc = np.clip(compl_vs_time_general(mag_parent, deltat, *coeffs),
                                     mmin, mag_parent)
                    else:
                        mc = mmin
                    m_offspr = inv_cdf_magnitude_trunc(u, b, mc, mmax)
             
            elif not model["magnitudetrunc"]:
                raise Exception("mag model untrunc not implemented")
            else:
                raise Exception("unknown mag model")
            
            u_r = uniform.rvs(size=num_offspr)
            u_theta = uniform.rvs(size=num_offspr) # loc=0., scale=2*np.pi, size=num_offspr)
            dm = mag4spat-mag_threshold
            if model["spatialpdf"] == 1:
                raise Exception("spatialpdf 1 not implemented")
            elif model["spatialpdf"] == 2:
                raise Exception("spatialpdf 2 not implemented")
            elif model["spatialpdf"] == 3:
                # f = (q - 1) / (D * np.exp(gamma * dm) * np.pi) * np.power(1 + r**2 / (D * np.exp(gamma * dm)), - q)
                # inverse cdf of the relative locations (f)
                # r = np.sqrt( params["D"]*((1-u)**(-1/(params["q"]-1))-1)/np.exp(-params["gamma"]*dm) )
                xsim, ysim = inv_cdf_space3(u_r, u_theta, dm, params["q"], params["D"])
            elif model["spatialpdf"] == 4:
                raise Exception("spatialpdf 4 not implemented")
            elif model["spatialpdf"] == 5:
                # f = (q - 1) / (D * np.exp(gamma * dm) * np.pi) * np.power(1 + r**2 / (D * np.exp(gamma * dm)), - q)
                # inverse cdf of the relative locations (f)
                # r = np.sqrt( params["D"]*((1-u)**(-1/(params["q"]-1))-1)/np.exp(-params["gamma"]*dm) )
                # xsim, ysim = inv_cdf_space5(u_r, u_theta, dm, params["q"], params["D"], params["gamma"])
                r_trunc = 1.
                xsim, ysim = inv_cdf_space5_trunc(u_r, u_theta, dm, params["q"],
                                                  params["D"], params["gamma"], r_trunc)
            x_offspr = x_par + xsim
            y_offspr = y_par + ysim
        
        else:
            x_offspr = np.array([])
            y_offspr = np.array([])
            m_offspr = np.array([])
        
        # out
            
        o = pd.DataFrame(zip(t_offspr, x_offspr, y_offspr, m_offspr-mag_threshold,
                             m_offspr, [None]*t_offspr.shape[0],
                             [False]*t_offspr.shape[0],
                             [par_main_mag]*t_offspr.shape[0], [mmax]*t_offspr.shape[0]),
                         columns = ['tt','xx','yy','mm','magnitude','geom',
                                    'isspontaneous','par_main_mag','check'])
        return o
    
    

    
    # @staticmethod
    # def calc_timedep_mc(params, gg_next_generation, gg_past_events,
    #                     mag_threshold, indtt):
    #     mcs = list()
    #     for g in gg_next_generation.itertuples(index=False): #.iterrows():
    #         ind = (gg_past_events["tt"] < g[indtt]) & \
    #               (g[indtt]-gg_past_events["tt"] < 1.) & \
    #               (gg_past_events["magnitude"] >= params["incompl_min_mag"])
    #         # ind = (gg_past_events["tt"] < g["tt"]) & \
    #         #       (g["tt"]-gg_past_events["tt"] < 1.) & \
    #         #       (gg_past_events["magnitude"] >= 6.)
    #         if np.any(ind):
    #             time = g[indtt] - gg_past_events.loc[ind]["tt"]
    #             mag_par = gg_past_events.loc[ind]["magnitude"]
    #             mc = params["c1"]*mag_par-params["c2"]-params["c3"]*np.log10(time)
    #             mc = np.clip(mc.values, mag_threshold, mag_par)
    #             mc_max = np.max(mc)
    #             mcs.append(mc_max)
    #         else:
    #             mcs.append(mag_threshold)
    #     return pd.Series(mcs, index=gg_next_generation.index)
    
    @staticmethod
    def calc_timedep_mc(params, next_generation_index, next_generation_tt,
                        past_events_tt, past_events_mag, mag_threshold):
        mcs = list()
        for i, g in enumerate(next_generation_tt):
            ind = (past_events_tt < g) & \
                  (g-past_events_tt < 1.) & \
                  (past_events_mag >= params["incompl_min_mag"])
            if np.any(ind):
                time = g - past_events_tt[ind]
                mag_par = past_events_mag[ind]
                # mc = params["c1"]*mag_par-params["c2"]-params["c3"]*np.log10(time)
                coeffs = [params["c1"], params["c2"], params["c3"]]                
                mc = compl_vs_time_general(mag_par, time, *coeffs)
                mc = np.clip(mc, mag_threshold, mag_par)
                mc_max = np.max(mc)
                mcs.append(mc_max)
            else:
                mcs.append(mag_threshold)
        return pd.Series(mcs, index=next_generation_index)
    

    @staticmethod
    def get_background_zhuang(fit, t_start, t_end, region_win_xy):
        '''
        Generate the background catalog with the estimated background intensity
        μ(x,y) (equation 11 in Zhuang et al. 2011)
        For each event in the background catalog, generate a random variable Ui 
        uniformly distributed in [0,1], accept it if Ui < ν φi dt/(t−t0), 
        where ν is as defined in eq (12) Zhuang et al. 2011
        t0 is the starting time of the catalog and t−t0 is the length
        of period of data fitted to the model.
        Randomly assign each selected event a new time uniformly distributed
        in [t,t+dt], and relocate each selected even by adding a 2D Gaussian
        random variable with a density Zdi, where Z is the kernel function
        used in estimating the background seismicity and di is the bandwidth
        corresponding to the selected event
        '''
        
        # get info from catalog and fit
        cat = fit.catalog.revents[fit.catalog.revents['flag'] == 1]
        pb = fit.pb[fit.catalog.revents['flag'] == 1]
        bwd = np.array(fit.bwd)[fit.catalog.revents['flag'] == 1]
        tt0 = fit.catalog.rtperiod['study_end'] - fit.catalog.rtperiod['study_start']
        
        # pick random events
        u = uniform.rvs(size=cat.shape[0])
        higher = u < fit.param['mu'] * pb * (t_end-t_start)/(tt0)
        
        # add 2D gaussian (https://upload.wikimedia.org/wikipedia/commons/a/a2/Cumulative_function_n_dimensional_Gaussians_12.2013.pdf)
        x_old = cat['xx'].iloc[higher].to_numpy()
        y_old = cat['yy'].iloc[higher].to_numpy()
        bwd = bwd[higher]
        theta = uniform.rvs(loc=0., scale=2*np.pi, size=bwd.shape[0])
        u = uniform.rvs(size=bwd.shape[0])
        r = np.sqrt( -2 * np.log(1-u) ) # inverse cdf of the relative locations (f)
        event_x = x_old + r*bwd*np.cos(theta)
        event_y = y_old + r*bwd*np.sin(theta)
        
        # filter for region win
        filt = np.array([Point(xxx, yyy).within(region_win_xy) for xxx, yyy in zip(event_x, event_y)])
        
        if np.sum(filt) == 0:
            backg = pd.DataFrame([], columns = ['tt','xx','yy','mm','magnitude','geometry','geom'])
        else:
            # random times
            times = uniform.rvs(loc=t_start, scale=(t_end-t_start), size=np.sum(filt))
            
            backg = pd.DataFrame(zip(times,
                                     event_x[filt],
                                     event_y[filt],
                                     cat['mm'].to_numpy()[higher][filt],
                                     cat['mm'].to_numpy()[higher][filt] + fit.catalog.mag_threshold,
                                     [None]*np.sum(filt),[None]*np.sum(filt)),
                             columns = ['tt','xx','yy','mm','magnitude','geometry','geom'])
            backg.sort_values(by=['tt'], inplace=True)
            backg.reset_index(inplace=True, drop=True)
        
            # heatmap, xedges, yedges = np.histogram2d(testx, testy, bins=50)
            # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            # plt.clf()
            # plt.imshow(heatmap.T, extent=extent, origin='lower')
            # plt.show()
        
        return backg



    @staticmethod
    def get_background_gridded(bkg, gen0, params, filters):
        '''
        background simulations
        gridded background
        '''
        
        if filters["t_start"] is None or filters["t_end"] is None:
            raise Exception("specify t_start and t_end in filters")
        if "mu" not in params.keys():
            raise Exception("mu not in params")
        
        t_start = filters["t_start"]
        t_end = filters["t_end"]
        study_length = t_end-t_start

        b = params['b']
        mag_threshold = mmin = params['min_mag']
        mmax = params['max_mag']
        
        yy, xx = np.meshgrid(bkg['lat'], bkg['lon'])
        xy = longlat2xy(xx, yy,
                        np.mean(gen0['longitude']),
                        np.mean(gen0['latitude']),
                        dist_unit="degree")
        bkg1 = deepcopy(bkg)
        bkg1["x"] = xy['x'][:,0]
        bkg1["y"] = xy['y'][0,:]
        bkg1["rated"] = params['mu'] * bkg1["rated"]

        region_win = longlat2xy(bkg1["bkg_region"].exterior.xy[0],
                                bkg1["bkg_region"].exterior.xy[1],
                                np.mean(gen0['longitude']),
                                np.mean(gen0['latitude']),
                                dist_unit="degree")
        region_win = get_region(region_win["x"], region_win["y"])
        
        # total integral of background seismicity in time window and area
         #TODO super slow
        integ0 = EtasMleBkg.calc_integ_bkg(bkg1, region_win,
                                           {'study_start': t_start,
                                            'study_end': t_end})
        print(integ0)
        
        # number time and location of events
        num_events = np.random.poisson(integ0)
        t_bkg = np.random.uniform(t_start, t_end, size=num_events)
        t_bkg.sort()
    
        # rupture sampler
        yy, xx = np.meshgrid(bkg1['y'], bkg1['x'])
        dx = (np.diff([np.min(bkg1['x']), np.max(bkg1['x'])]) / (bkg1['x'].shape[0]-1))[0]
        dy = (np.diff([np.min(bkg1['y']), np.max(bkg1['y'])]) / (bkg1['y'].shape[0]-1))[0]
        distr = list()
        for i in range(0,bkg1['rated'].shape[0]):
            for j in range(0,bkg1['rated'].shape[1]):
                if Point(xx[i,j], yy[i,j]).within(region_win):
                    prob = bkg1['rated'][i,j] * dx * dy * study_length / \
                           integ0
                    distr.append((prob, (i,j)))
        nthRupRandomSampler = PMF(distr)
        ids = nthRupRandomSampler.sample_pairs(num_events)
        
        # random uniform distribution within spatial bin
        _plus_x = np.random.uniform(-dx/2, dx/2, size=num_events)
        _plus_y = np.random.uniform(-dy/2, dy/2, size=num_events)
        x_bkg = np.array([ bkg1['x'][i[1][0]] for i in ids ]) + _plus_x
        y_bkg = np.array([ bkg1['y'][i[1][1]] for i in ids ]) + _plus_y 
        
        # magnitude
        u = uniform.rvs(size=num_events)
        m_bkg = inv_cdf_magnitude_trunc(u, b, mmin, mmax)

        # output
        o = pd.DataFrame(zip(t_bkg, x_bkg, y_bkg, m_bkg-mag_threshold,
                             m_bkg, [None]*num_events, ["bkg"]*num_events,
                             [None]*num_events, [None]*num_events),
                         columns = ['tt','xx','yy','mm','magnitude','geom',
                                    'isspontaneous','par_main_mag','check'])
        return o


