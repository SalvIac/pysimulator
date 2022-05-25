# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import datetime
import numpy as np
from pyetas.utils import get_region
# from pysimulator.etas_simulator import EtasSimulator
from pysimulator.etas_simulator_slow import EtasSimulator
from pysimulator.custom_catalog import CustomCatalog
from pysimulator.rupture_builder import RuptureBuilder
from pysimulator.support_functions import (plot_time_etas_single,
                                           plot_space_etas_single_1,
                                           plot_time_etas_bulk,
                                           plot_space_etas_bulk)


if __name__ == "__main__":
    
    #%% settings
    # for theory and a few more details see Iacoletti et al. (2022)
    # https://pubs.geoscienceworld.org/ssa/srl/article-abstract/doi/10.1785/0220210134/611678/Validation-of-the-Epidemic-Type-Aftershock?redirectedFrom=fulltext
    
    # other ETAS parameter (values below are examples only)  
    params = {
            "mu": 1.,
            "alpha": 1.6,
            "A": 0.2,
            "p": 1.2,
            "c": 0.001,
            "ta": 5*365,
            "q": 2.,
            "D": 0.001,
            "gamma": 0.8,
            "b": 1.,
            "min_mag": 3.,
            "max_mag": 7.2,
            "c1": 1.,
            "c2": 4.5,
            "c3": 0.75,
            "incompl_min_mag": 6,
            "gr_incompl": True,
            }
    
    # truncated time pdf, truncated magnitude pdf
    # spatial pdf 5 (see Zhuang et al. 2011), magnitude incompleteness considered
    model = "timetrunc:True_magnitudetrunc:True_spatialpdf:5_incompletess:True"
    # 100 realizations, use multiprocessing with 4 cores
    options = {"num_realization": 1000, "multiprocessing": True, "cores": 4}
    
    # 1-year long simulation
    days_ahead = 365


    #%% time window filters
    
    # time-windows (simulation start and simulation end)
    sim_start = datetime.datetime(2022, 2, 22, 20, 2)
    sim_end = sim_start + datetime.timedelta(days=days_ahead)
    
    filters = {"sim_start": sim_start,
               "sim_end": sim_end}
    print(filters)
    
    
    #%% gridded background seismicity

    # polygon region for background events only
    bkg_region = get_region([-1.,-1.,1.,1.], [-1.,1.,1.,-1.]) # polygon region
    
    # define grid between -1deg and 1deg for longitudes and latitudes
    lonsx = np.arange(-1., 1., .1)+.05 # longitude center bins (e.g., 0.05 is the bin covering range [0,0.1])
    latsy = np.arange(-1., 1., .1)+.05 # latitude  center bins (e.g., 0.05 is the bin covering range [0,0.1])
    
    # simple gridded background daily rates per unit area [constant 0.1 events/(yr*deg^2)]
    rated = 0.1 * np.ones((lonsx.shape[0], latsy.shape[0]))
    
    # # more complex background distribution
    # from scipy.stats import multivariate_normal
    # x, y = np.meshgrid(lonsx, latsy)
    # pos = np.dstack((x, y))
    # rv = multivariate_normal([0.5, 0.5], [[1, 0], [0, 1]])
    # rated = rv.pdf(pos)
    
    # define input for EtasSimulator
    bkg = {"type": "gridded",
            'lon': lonsx, 'lat': latsy,
            'rated': rated,
            "bkg_region": bkg_region}
    options["background"] = bkg
    

    #%% example inputs catalog
    # magnitude 3. earthquake occurred at sim_start in (lon,lat,depth)=(0,0,10)

    prev_datetimes = [sim_start]
    prev_ruptures = [RuptureBuilder.init_point(mag=3., lon=0., lat=0., depth=10.,
                                               rake=None, strike=None, dip=None)]
    catalogs_start = CustomCatalog(prev_datetimes, prev_ruptures)
    print(catalogs_start)
    

    #%% actual simulations (ETAS)
    
    etas_sim = EtasSimulator(params, catalogs_start, model,
                             simul_options=options, filters=filters)
    simulations = etas_sim.simulate()
    
    
    #%% some plots
    
    # time plot first realization
    plot_time_etas_single(simulations[0], mag_threshold=3.)
    
    # simple space plot first realization
    plot_space_etas_single_1(simulations[0], mag_threshold=3.)
    
    # summary time plot of all realizations
    plot_time_etas_bulk(simulations, mag_threshold=3.)
    
    # summary space plot of all realizations
    plot_space_etas_bulk(simulations, mag_threshold=3., show=True)
      
    
    #%% same plots only considering background seismicity
    
    # filter background events
    simulations_bkg = [cat.filter([i == "bkg" for i in cat.catalog["mainshock"]]) for cat in simulations]
    
    # time plot first realization
    plot_time_etas_single(simulations_bkg[0], mag_threshold=3.)
    
    # simple space plot first realization
    plot_space_etas_single_1(simulations_bkg[0], mag_threshold=3.)
    
    # summary time plot of all realizations
    plot_time_etas_bulk(simulations_bkg, mag_threshold=3.)
    
    # summary space plot of all realizations
    plot_space_etas_bulk(simulations_bkg, mag_threshold=3., show=True)
       
