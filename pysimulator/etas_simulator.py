# -*- coding: utf-8 -*-
# pyetas
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
Module :mod:`pyrisk.simulator.etas_simulator` defines
:class:`EtasSimulator`.
"""

import os
import datetime
import warnings
from copy import copy, deepcopy
from itertools import compress

import numpy as np
import pandas as pd
from scipy.stats import uniform, poisson
from shapely.geometry import Point
from tqdm import tqdm

from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.pmf import PMF

from pyetas.utils import xy2longlat, longlat2xy, get_region
from pyetas.etas8p.etas_mle_bkg import EtasMleBkg
from pysimulator.simulation_functions import (inv_cdf_magnitude_trunc,
                                              inv_cdf_time,
                                              inv_cdf_time_trunc,
                                              inv_cdf_space5,
                                              inv_cdf_space5_trunc,
                                              inv_cdf_space3,
                                              compl_vs_time_p16,
                                              compl_vs_time_hs06,
                                              compl_vs_time_general)
from pysimulator.custom_catalog import CustomCatalog
from pysimulator.rupture_builder import RuptureBuilder
from pysimulator.etas_simulator_slow import (concat, concat_list, filter_dict,
                                             filter_dict, sort_by)
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
    def __init__(self, params, input_catalog, model=None, fault_mode=True,
                 nodal_planes_distr=[], depth_distr=[], filters={},
                 simul_options={}):

        # etas model
        self.params = params
        self.model = self._model2dict(model)
        self.fault_mode = fault_mode

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
        self.seed = 42
        
    
    
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

        # in single_event mode, we have one single input_catalog, to be used 
        # several times to generate aftershocks
        if self.mode == "single_event":
            cat = self.input_catalog[0]
            if len(cat.catalog["datetime"]) != 0:
                cat.process_catalog_4etas(self.params["min_mag"])
            _gen0 = cat.catalog
            if not self.fault_mode:
                _gen0["geom"] = [None]*cat.get_num_events()
        
            filters = deepcopy(self.filters)
            if (filters["sim_start"] is not None) and \
               (filters["sim_end"] is not None) and \
               len(cat.catalog['datetime']) != 0:
                tdt = cat.datetime2tdt(filters["sim_start"],
                                       filters["sim_end"],
                                       pd.Series(cat.catalog['datetime']).min())
                filters["t_start"] = tdt[0]
                filters["t_end"] = tdt[1]
            
            background = None
            if self.simul_options['background'] is not None:
                background = EtasSimulator.integrate_background_gridded(
                                    self.simul_options['background'], _gen0,
                                    self.params, filters)
                print("background integral:", background["integ0"], "events")
        
        inds = self.slice2inds()
        for i, ind in tqdm(inds):
            # in stoch_catalog mode, we have different input_catalogs
            if self.mode == "stoch_catalog":
                cat = self.input_catalog[i]
                if len(cat.catalog["datetime"]) != 0:
                    cat.process_catalog_4etas(self.params["min_mag"])
                _gen0 = cat.catalog
                if not self.fault_mode:
                    _gen0["geom"] = [None]*cat.get_num_events()
            
                filters = deepcopy(self.filters)
                if (filters["sim_start"] is not None) and \
                   (filters["sim_end"] is not None) and \
                   len(cat.catalog['datetime']) != 0:
                    tdt = cat.datetime2tdt(filters["sim_start"],
                                           filters["sim_end"],
                                           pd.Series(cat.catalog['datetime']).min())
                    filters["t_start"] = tdt[0]
                    filters["t_end"] = tdt[1]
                
                background = None
                if self.simul_options['background'] is not None:
                    background = EtasSimulator.integrate_background_gridded(
                                        self.simul_options['background'], _gen0,
                                        self.params, filters)
                    print("background integral:", background["integ0"], "events")

            dic = {'gen0': deepcopy(_gen0),
                   'params': self.params,
                   'model': self.model,
                   'only_first_generation': self.simul_options['only_first_generation'],
                   'background': background,
                   'seed': self.seed*(ind+1),
                   'filters': filters}
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
        else: #TODO I don't like that I had to copy-paste code from above here to save RAM
            print("simulate etas model")
            
            if self.mode == "single_event":
                cat = self.input_catalog[0]
                if len(cat.catalog["datetime"]) != 0:
                    cat.process_catalog_4etas(self.params["min_mag"])
                _gen0 = cat.catalog
                if not self.fault_mode:
                    _gen0["geom"] = [None]*cat.get_num_events()
            
                filters = deepcopy(self.filters)
                if (filters["sim_start"] is not None) and \
                   (filters["sim_end"] is not None) and \
                   len(cat.catalog['datetime']) != 0:
                    tdt = cat.datetime2tdt(filters["sim_start"],
                                           filters["sim_end"],
                                           pd.Series(cat.catalog['datetime']).min())
                    filters["t_start"] = tdt[0]
                    filters["t_end"] = tdt[1]
                
                background = None
                if self.simul_options['background'] is not None:
                    background = EtasSimulator.integrate_background_gridded(
                                        self.simul_options['background'], _gen0,
                                        self.params, filters)
                    print("background integral:", background["integ0"], "events")
            
            out = list()
            inds = self.slice2inds()
            for i, ind in tqdm(inds):
                # in stoch_catalog mode, we have different input_catalogs
                if self.mode == "stoch_catalog":
                    cat = self.input_catalog[i]
                    if len(cat.catalog["datetime"]) != 0:
                        cat.process_catalog_4etas(self.params["min_mag"])
                    _gen0 = cat.catalog
                    if not self.fault_mode:
                        _gen0["geom"] = [None]*cat.get_num_events()
                
                    filters = deepcopy(self.filters)
                    if (filters["sim_start"] is not None) and \
                       (filters["sim_end"] is not None) and \
                       len(cat.catalog['datetime']) != 0:
                        tdt = cat.datetime2tdt(filters["sim_start"],
                                               filters["sim_end"],
                                               pd.Series(cat.catalog['datetime']).min())
                        filters["t_start"] = tdt[0]
                        filters["t_end"] = tdt[1]
                    
                    background = None
                    if self.simul_options['background'] is not None:
                        background = EtasSimulator.integrate_background_gridded(
                                            self.simul_options['background'], _gen0,
                                            self.params, filters)
                        print("background integral:", background["integ0"], "events")
                    
                dic = {'gen0': deepcopy(_gen0),
                       'params': self.params,
                       'model': self.model,
                       'only_first_generation': self.simul_options['only_first_generation'],
                       'background': background,
                       'seed': self.seed*(ind+1),
                       'filters': filters}
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
            depth = self.get_depth(o["tt"].shape[0])
            nodpl = self.get_nodalplane(o["tt"].shape[0])
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
        for i in range(0, o["tt"].shape[0]):
            date_times.append(o["datetime"][i])
            if (not o["isspontaneous"][i]) or (o["isspontaneous"][i] == "bkg"):
                ruptures.append( RuptureBuilder.init_point(
                                                o["magnitude"][i],
                                                o["longitude"][i],
                                                o["latitude"][i],
                                                depth[i],
                                                nodpl[i].rake,
                                                nodpl[i].strike,
                                                nodpl[i].dip) )
            else:
                ruptures.append( o["rupture"][i] )
        return CustomCatalog(date_times, ruptures,
                             mainshock=o["isspontaneous"])
        
    
    
    
    def get_depth(self, num):
        return self.depth_distr.sample(num)

    
    def get_nodalplane(self, num):
        return self.nodal_planes_distr.sample(num)


    def set_seed(self, seed):
        self.seed = seed

        
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
        only_first_generation = inputs['only_first_generation']
        background = inputs['background']
        seed = inputs['seed']
        filters = inputs['filters']
        
        # for reproducibility
        np.random.seed(seed=seed)
        
        if background is not None:
            if background["bkg1"]["type"] == "gridded":
                bkgr = EtasSimulator.get_background_gridded(background, params,
                                                            filters)
            elif background["bkg1"]["type"] == "zhuang":
                raise Exception("get_background_zhuang not yet complete.")
                bkgr = EtasSimulator.get_background_zhuang(fit, t, tdt, buffer_region_xy)
            gen01 = concat(gen0, bkgr)
        else:
            gen01 = gen0
        
        if gen01["tt"].shape[0] == 0:
            return gen01
        gg = [gen01] # list with each generation, regardless the parent event
        
        timedep_mc = None
        if model["incompletess"]:
            timedep_mc = EtasSimulator.calc_timedep_mc(params,
                                                       # gg[-1].index,
                                                       gg[-1]["tt"],
                                                       gg[-1]["tt"],
                                                       gg[-1]["magnitude"])
        gl = EtasSimulator.get_following_generation(gg[-1], params, model,
                                                    timedep_mc, filters)
        if len(gl) != 0:
            gg.append(concat_list(gl))
        
        if not only_first_generation:
            l = 1
            while len(gl) != 0:
                if model["incompletess"]:
                    temp = concat_list(gg, ["tt", "magnitude"])
                    timedep_mc = EtasSimulator.calc_timedep_mc(params,
                                                               gg[-1]["tt"],
                                                               temp["tt"],
                                                               temp["magnitude"])
                gl = EtasSimulator.get_following_generation(gg[-1], params, model,
                                                            timedep_mc, filters)
                if len(gl) != 0:
                    gg.append(concat_list(gl))
                    l += 1
        stoch_catalog = concat_list(gg)
        
        # convert x y in lon lat
        proj = xy2longlat(stoch_catalog['xx'],
                          stoch_catalog['yy'],
                          np.mean(gen0['longitude']),
                          np.mean(gen0['latitude']),
                          dist_unit="degree")
        stoch_catalog['longitude'] = proj['long']
        stoch_catalog['latitude'] = proj['lat']
        
        # convert in pandas dataframe only for duplicate removal
        stcat_df = pd.DataFrame(stoch_catalog)
        filt = ~stcat_df.duplicated(subset=['tt', 'xx', 'yy', 'mm']).values
        stoch_catalog = filter_dict(stoch_catalog, filt)
        
        # sort by time
        stoch_catalog = sort_by(stoch_catalog, 'tt')
        # convert tt (days) in datetime (no need to filter here)
        stoch_catalog["datetime"] = stoch_catalog["datetime"][0] - \
                                    datetime.timedelta(days=stoch_catalog["tt"][0]) + \
                                    np.array([datetime.timedelta(days=days) for days in stoch_catalog["tt"]])

        # filter magnitude
        stoch_catalog = EtasSimulator.filter_events_mag(stoch_catalog,
                                                        filters["min_mag"])
        # filters time (no need to do it here anymore, moved within the loop)
        stoch_catalog = EtasSimulator.filter_events_time(stoch_catalog,
                                                         filters["t_start"],
                                                         filters["t_end"])
        # filter space
        stoch_catalog = EtasSimulator.filter_events_space(stoch_catalog,
                                                          filters["region"])
        return stoch_catalog



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
        mag_par = list()
        mag4spat = list()
        time_par = list()
        x_par = list()
        y_par = list()
        points = list()
        timedep_mc_par = list()
        main_mag_ref = list()
        wc = WC1994()
        for par, geom in enumerate(gen['geom']):
            if geom is None:
                mag_par.append(gen['magnitude'][par])
                mag4spat.append(gen['magnitude'][par])
                time_par.append(gen['tt'][par])
                x_par.append(gen['xx'][par])
                y_par.append(gen['yy'][par])
                points.append(1)
                if timedep_mc is not None:
                    timedep_mc_par.append(timedep_mc[par])
                else:
                    timedep_mc_par.append(None)
                main_mag_ref.append(gen["main_mag_ref"][par])
            else:
                for i in range(0, geom['x'].shape[0]):
                    mag_par.append(gen['magnitude'][par])
                    time_par.append(gen['tt'][par])
                    x_par.append(geom['x'][i])
                    y_par.append(geom['y'][i])
                    points.append(geom['x'].shape[0])
                    if timedep_mc is not None:
                        timedep_mc_par.append(timedep_mc[par])
                    else:
                        timedep_mc_par.append(None)
                    # mag4spat.append(mag_par[-1])
                    mag4spat.append( np.max([ mag_threshold,
                                       wc.get_median_mag(wc.get_median_area(mag_par[-1], None)/points[-1], None) ]))
                    main_mag_ref.append(gen["main_mag_ref"][par])
                    
        # productivity (corrected by the number of points UCERF3)
        if timedep_mc is None:
            km = (params["A"] / np.array(points)) * np.exp(params["alpha"] * \
                                              (np.array(mag_par)-mag_threshold))
        else:
            km = (params["A"] / np.array(points)) * np.exp(params["alpha"] * \
                                              (np.array(mag_par)-np.array(timedep_mc_par)))

        # random number of offspring events
        ni = poisson.rvs(km)
        if isinstance(ni, int):
            ni = np.array([ni])
    
        ol = list() # list of offspring for each event of the generation l
        for par in range(0, ni.shape[0]):
            if ni[par] > 0:
                o = EtasSimulator.generate_offspring(params, model, ni[par],
                                                     time_par[par],
                                                     x_par[par], y_par[par],
                                                     mag_par[par],
                                                     mag4spat[par],
                                                     main_mag_ref[par])
                # preliminary filtering in time avoids unnecessary computational burden
                o = EtasSimulator.filter_events_time(o, t_start, t_end)
                ol.append(o)
        return ol
    
    
    
    @staticmethod
    def filter_events_space(o, region=None):
        if region is not None:
            filt = np.array([Point(xxx, yyy).within(region) for xxx, yyy 
                             in zip(o['longitude'], o['latitude'])])
            if len(filt) == 0:
                return o
            o = filter_dict(o, filt)
        return o
    
    
    @staticmethod
    def filter_events_mag(o, mag_min=None):
        if mag_min is not None:
            filt = o['magnitude'] >= mag_min
            o = filter_dict(o, filt)
        return o
    
    
    @staticmethod
    def filter_events_time(o, t_start=None, t_end=None):
        if (t_start is not None) and (t_end is not None):
            filt = (o['tt'] >= t_start) & (o['tt'] <= t_end)
            o = filter_dict(o, filt)
        else:
            if (t_start is not None):
                filt = (o['tt'] >= t_start)
                o = filter_dict(o, filt)
            elif (t_end is not None):
                filt = (o['tt'] <= t_end)
                o = filter_dict(o, filt)
        return o
    
    
    
    @staticmethod
    def generate_offspring(params, model, num_offspr, time_par, x_par, y_par,
                           mag_par, mag4spat, main_mag_ref=None):
        '''
        given one parent event, generate offspring
        params: parameters of ETAS, dict
        model: ETAS model specs, dict
        num_offspr: number of offspring, int
        time_par: time parent, float
        x_par: x parent, float
        y_par: y parent, float
        mag_par: magnitude parent, float
        mag4spat: scaled magnitude to use for spatial pdf, float
        main_mag_ref: keeps track of the original mainshock that generated all 
                      the events
        if necessary, this could be subdivided in generate_time_offspring,
        generate_space_offspring, generate_mag_offspring and track_info
        '''
        b = params['b']
        mag_threshold = params['min_mag']
        mag_max = params['max_mag']
        # this tracks the magnitude of the mainshock causing all the event tree
        if main_mag_ref is None:
            main_mag_ref = mag_par
        
        #######################################################################
        
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
        
        #######################################################################
        
        if num_offspr > 0:
            # b = (1. / gl['mm'].mean())/np.log(10.)
            mmax = min(main_mag_ref, mag_max)
            mmin = mag_threshold
            # m_offspr = mmin + (np.log10(1.-u*(1.-10.**(-b*(mmax-mmin)))))/(-b)
            u = uniform.rvs(size=num_offspr)
            if model["magnitudetrunc"]:
                if not model["incompletess"]:
                    m_offspr = inv_cdf_magnitude_trunc(u, b, mmin, mmax)
                elif model["incompletess"]:
                    coeffs = [params["c1"], params["c2"], params["c3"]]
                    if mag_par >= params["incompl_min_mag"] and params["gr_incompl"]:
                        mc = np.clip(compl_vs_time_general(mag_par, deltat, *coeffs),
                                      mmin, mag_par)
                    else:
                        mc = mmin
                    # mc = mmin
                    m_offspr = inv_cdf_magnitude_trunc(u, b, mc, mmax)
             
            elif not model["magnitudetrunc"]:
                raise Exception("mag model untrunc not implemented")
            else:
                raise Exception("unknown mag model")
        
            ###################################################################
        
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
        o = {'tt': t_offspr,
             'xx': x_offspr,
             'yy': y_offspr,
             'mm': m_offspr-mag_threshold,
             'magnitude': m_offspr,
             'geom': [None]*t_offspr.shape[0],
             'isspontaneous': [False]*t_offspr.shape[0],
             'main_mag_ref': [main_mag_ref]*t_offspr.shape[0]}
        return o
    
    
    
    
    
    

    @staticmethod
    def calc_timedep_mc(params, next_generation_tt, past_events_tt,
                        past_events_mag):
        '''
        params: dict
        next_generation_tt: numpy nparray
        past_events_tt: numpy nparray
        past_events_mag: numpy nparray
        '''
        mag_threshold = params["min_mag"]
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
        return np.array(mcs)
    

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
    def integrate_background_gridded(bkg, gen0, params, filters):
        if filters["t_start"] is None or filters["t_end"] is None:
            raise Exception("specify t_start and t_end in filters")
        if "mu" not in params.keys():
            raise Exception("mu not in params")
        
        t_start = filters["t_start"]
        t_end = filters["t_end"]

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
        return {"bkg1": bkg1, "region_win": region_win, "integ0": integ0}



    @staticmethod
    def get_background_gridded(bkg, params, filters):
        '''
        background simulations with gridded background
        '''
        
        bkg1 = bkg["bkg1"]
        region_win = bkg["region_win"]
        integ0 = bkg["integ0"]
        
        t_start = filters["t_start"]
        t_end = filters["t_end"]
        study_length = t_end-t_start

        b = params['b']
        mag_threshold = mmin = params['min_mag']
        mmax = params['max_mag']
        
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
        o = {"tt": t_bkg,
             "xx": x_bkg,
             "yy": y_bkg,
             "mm": m_bkg-mag_threshold,
             "magnitude": m_bkg,
             "geom": [None]*num_events,
             "isspontaneous": ["bkg"]*num_events,
             'main_mag_ref': [None]*num_events}
        return o


