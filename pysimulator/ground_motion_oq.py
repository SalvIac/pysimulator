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
import scipy.stats
from tqdm import tqdm
import toml

from openquake.baselib.general import AccumDict
from openquake.hazardlib.const import StdDev
from openquake.hazardlib.imt import from_string
from openquake.hazardlib.calc.gmf import to_imt_unit_values, rvs
from openquake.hazardlib.site import SiteCollection
from openquake.hazardlib.gsim.base import ContextMaker, FarAwayRupture
from openquake.commonlib import oqvalidation
from openquake.calculators import base
from openquake.hazardlib import valid
from openquake.hazardlib.gsim.base import registry

from pyspatialpca.spatial_pca_residuals import SpatialPcaResiduals
from pysimulator.ground_motion_container import (GmfGroupContainer,
                                                 GmfCatalogContainer)
from myutils.run_multiprocess import run_multiprocess

F32 = np.float32


class GroundMotion():
    
    multiprocessing = False
    cores = 4
    seed = 42
    T_sim = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
                       0.3,  0.4,  0.5, 0.75, 1.   , 1.5, 2.  , 3. , 4.  , 5.])
    
    def __init__(self, params, lons, lats, T_sim=None, vs30=None):
        
        self.lons = lons
        self.lats = lats
        if T_sim is not None:
            self.T_sim = T_sim
        params['intensity_measure_types'] = self.get_imts(self.T_sim) #'PGA, SA(0.02), SA(0.03), SA(0.05), SA(0.075), SA(0.1), SA(0.15), SA(0.2), SA(0.25), SA(0.3), SA(0.4), SA(0.5), SA(0.75), SA(1.0), SA(1.5), SA(2.0), SA(3.0), SA(4.0), SA(5.0)',
        self.params = params
         
        # general oq parameters
        oqvalidation.OqParam.calculation_mode.validator.choices = tuple(
                                                            base.calculators)
        self.oq = oqvalidation.OqParam(**params)

        # sites
        depths = None
        req_site_params = ('z2pt5', 'z1pt0', 'backarc')
        self.sitecol = SiteCollection.from_points(lons, lats, depths, self.oq,
                                                  req_site_params)
        if vs30 is not None:
            self.sitecol.array["vs30"] = np.array(vs30)
        
        # ground motion model
        value = self.oq.gsim
        # #TODO in theory this should work but for some reason it calls the
        # super init method only (can it be the python version?).
        # copy-pasting some of the code inside valid.gim here works
        # self.gsims = [valid.gsim(value)]
        value = valid.to_toml(value)  # convert to TOML
        [(gsim_name, kwargs)] = toml.loads(value).items()
        kwargs = valid._fix_toml(kwargs)
        try:
            gsim_class = registry[gsim_name]
        except KeyError:
            raise ValueError('Unknown GSIM: %s' % gsim_name)
        gs = gsim_class(**kwargs)
        gs._toml = '\n'.join(line.strip() for line in value.splitlines())
        self.gsims = [gs]
        
        # context maker
        self.cmaker = ContextMaker('TRT', self.gsims, dict(imtls=self.oq.imtls,
                                    truncation_level=self.oq.truncation_level))


    
    def get_norm_res(self, nsims, nPCs=5):        
        # spatial pca simulate normalized residuals
        # nPCs = 5  # number of principal components to be used in simulation
        spr = SpatialPcaResiduals.from_lonlat(self.lons, self.lats)
        sim_results = spr.simulate_residuals(self.T_sim, nsims, nPCs)
        return sim_results


    @classmethod
    def get_imts(cls, T_sim):
        out = ""
        for t in T_sim:
            if t == 0.01:
                out += "PGA, "
            else:
                out += "SA({}), ".format(t)
        return out[:-2]


    def compute(self, catalogs, spatialpca=True):
        #TODO
        # realizations of the gmpe logic tree
        
        # count number of events (i.e., number of simulations of residuals)
        if spatialpca:
            nsims = 0 # in this case nsims is the number of events
            for catalog in catalogs:
                nsims += catalog.get_num_events()
            print("number of normalized residuals: "+str(nsims))
            # this is for a faster simulations
            norm_res = self.get_norm_res(nsims)
            
            # this is to apply truncation (#TODO find a better way! this is superslow)
            rep = [np.any(norm_res[:,:,i]>self.oq.truncation_level) or
                   np.any(norm_res[:,:,i]<-self.oq.truncation_level) 
                   for i in range(0, nsims)]
            while np.any(rep):
                norm_res2 = self.get_norm_res(np.sum(rep))
                norm_res[:,:,rep] = norm_res2
                rep = [np.any(norm_res[:,:,i]>self.oq.truncation_level) or
                       np.any(norm_res[:,:,i]<-self.oq.truncation_level) 
                       for i in range(0, nsims)]
            ###################################### this is to apply truncation            
            
            print(norm_res.shape)
            print(self.T_sim)
            print("done simulating normalized residuals")

        # gcc = list()
        # c = 0 # keep track of events (to get the normalized residuals)
        # for catalog in tqdm(catalogs):
        #     gmfs = list()
        #     datetimes = list()
        #     mainshock_flags = list()
        #     for r, rup in enumerate(catalog.catalog["rupture"]):
        #         gmf_computer = GmfComputerMod(rupture=rup,
        #                                       sitecol=self.sitecol, 
        #                                       cmaker=self.cmaker)
        #         [mean_stds] = self.cmaker.get_mean_stds([gmf_computer.ctx],
        #                                                 StdDev.EVENT)
        #         if spatialpca:
        #             res = gmf_computer.compute(self.gsims[0], 1, mean_stds,
        #                                        norm_res[:,:,c])
        #         else:
        #             res = gmf_computer.compute(self.gsims[0], 1, mean_stds)
        #         c += 1

        #         gmfs.append(res[:,:,0])
        #         datetimes.append(catalog.catalog["datetime"][r])
        #         mainshock_flags.append(catalog.catalog["mainshock"][r])
        #     gcc.append( GmfCatalogContainer(gmfs, datetimes,
        #                                           mainshock=mainshock_flags) )

        c = 0
        inputs = list()
        for i, catalog in enumerate(catalogs):
            inputs.append([self.seed+i, self.sitecol, self.cmaker, self.gsims,
                           catalog, spatialpca, 
                           norm_res[:,:,slice(c, c+len(catalog.catalog["datetime"]))]] )
            c += len(catalog.catalog["datetime"])
            
        # simulate ground motion for each catalog
        print("start gmf simulations")
        if self.multiprocessing:
            gcc = run_multiprocess(self.simulation_single_gmf, inputs,
                                   self.cores)
        else:
            gcc = list()
            for inpu in tqdm(inputs):
                gcc.append( self.simulation_single_gmf(inpu) )
        
        return GmfGroupContainer(gcc, self.sitecol, self.oq.imtls)
            
        

        
    @classmethod
    def simulation_single_gmf(cls, inputs):
        
        seed = inputs[0]
        np.random.seed(seed)
        sitecol = inputs[1]
        cmaker = inputs[2]
        gsims = inputs[3]
        catalog = inputs[4]
        spatialpca = inputs[5]
        norm_res = inputs[6]
        num_sites = len(sitecol.complete.array)
        
        computers = list()
        for r, rup in enumerate(catalog.catalog["rupture"]):
            gmf_computer = GmfComputerMod(rupture=rup,
                                          sitecol=sitecol, 
                                          cmaker=cmaker)
            computers.append(gmf_computer)
        mean_stds = cmaker.get_mean_stds([gmf_computer.ctx for gmf_computer in computers])
        c = 0 # keep track of events (to get the normalized residuals)
        gmfs = list()
        datetimes = list()
        mainshock_flags = list()
        for r, rup in enumerate(catalog.catalog["rupture"]):
            gmf_computer = computers[r]
            if spatialpca:
                res = gmf_computer.compute(gsims[0], 1,
                                           mean_stds[:,:,:,r*num_sites:(r+1)*num_sites],
                                           norm_res[:,:,c])
            else:
                res = gmf_computer.compute(gsims[0], 1,
                                           mean_stds[:,:,:,r*num_sites:(r+1)*num_sites])
            c += 1
            # #### check
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(self.T_sim, res[:,0,0], label="pca")
            # plt.legend()
            # plt.show()
            gmfs.append(res[:,:,0])
            datetimes.append(catalog.catalog["datetime"][r])
            if "mainshock" in catalog.catalog.keys():
                mainshock_flags.append(catalog.catalog["mainshock"][r])
        if "mainshock" in catalog.catalog.keys():
            return GmfCatalogContainer(gmfs, datetimes, mainshock=mainshock_flags)
        else:
            return GmfCatalogContainer(gmfs, datetimes)


    def set_seed(self, seed):
        np.random.seed(seed)
        self.seed = seed








class GmfComputerMod(object):
    """
    Inspired by GmfComputer in the Openquake engine (adding spatial PCA)
    
    Given an earthquake rupture, the ground motion field computer computes
    ground shaking over a set of sites, by randomly sampling a ground
    shaking intensity model.

    :param rupture:
        Rupture to calculate ground motion fields radiated from.

    :param :class:`openquake.hazardlib.site.SiteCollection` sitecol:
        a complete SiteCollection

    :param imts:
        a sorted list of Intensity Measure Type strings

    :param cmaker:
        a :class:`openquake.hazardlib.gsim.base.ContextMaker` instance

    :param truncation_level:
        Float, number of standard deviations for truncation of the intensity
        distribution, or ``None``.

    :param correlation_model:
        Instance of correlation model object. See
        :mod:`openquake.hazardlib.correlation`. Can be ``None``, in which
        case non-correlated ground motion fields are calculated.
        Correlation model is not used if ``truncation_level`` is zero.

    :param amplifier:
        None or an instance of Amplifier
    """
    # The GmfComputer is called from the OpenQuake Engine. In that case
    # the rupture is an higher level containing a
    # :class:`openquake.hazardlib.source.rupture.Rupture` instance as an
    # attribute. Then the `.compute(gsim, num_events, ms)` method is called and
    # a matrix of size (I, N, E) is returned, where I is the number of
    # IMTs, N the number of affected sites and E the number of events. The
    def __init__(self, rupture, sitecol, cmaker, correlation_model=None,
                 amplifier=None, sec_perils=()):
        if len(sitecol) == 0:
            raise ValueError('No sites')
        elif len(cmaker.imtls) == 0:
            raise ValueError('No IMTs')
        elif len(cmaker.gsims) == 0:
            raise ValueError('No GSIMs')
        self.cmaker = cmaker
        self.imts = [from_string(imt) for imt in cmaker.imtls]
        self.cmaker = cmaker
        self.gsims = sorted(cmaker.gsims)
        self.correlation_model = correlation_model
        self.amplifier = amplifier
        self.sec_perils = sec_perils
        # `rupture` is an EBRupture instance in the engine
        if hasattr(rupture, 'source_id'):
            self.ebrupture = rupture
            self.source_id = rupture.source_id  # the underlying source
            rupture = rupture.rupture  # the underlying rupture
        else:  # in the hazardlib tests
            self.source_id = '?'
        self.seed = rupture.rup_id
        ctxs = cmaker.get_ctxs([rupture], sitecol, self.source_id)
        if not ctxs:
            raise FarAwayRupture
        self.ctx = ctxs[0]
        if correlation_model:  # store the filtered sitecol
            self.sites = sitecol.complete.filtered(self.ctx.sids)
        if cmaker.trunclevel is None:
            self.distribution = scipy.stats.norm()
        elif cmaker.trunclevel == 0:
            self.distribution = None
        else:
            assert cmaker.trunclevel > 0, cmaker.trunclevel
            self.distribution = scipy.stats.truncnorm(
                - cmaker.trunclevel, cmaker.trunclevel)

    # def compute_all(self, min_iml, rlzs_by_gsim, sig_eps=None):
    #     """
    #     :returns: (dict with fields eid, sid, gmv_...), dt
    #     """
    #     t0 = time.time()
    #     sids = self.sids
    #     eids_by_rlz = self.ebrupture.get_eids_by_rlz(rlzs_by_gsim)
    #     mag = self.ebrupture.rupture.mag
    #     data = AccumDict(accum=[])
    #     mean_stds = self.cmaker.get_mean_stds([self.ctx], StdDev.EVENT)
    #     # G arrays of shape (O, N, M)
    #     for g, (gs, rlzs) in enumerate(rlzs_by_gsim.items()):
    #         num_events = sum(len(eids_by_rlz[rlz]) for rlz in rlzs)
    #         if num_events == 0:  # it may happen
    #             continue
    #         # NB: the trick for performance is to keep the call to
    #         # .compute outside of the loop over the realizations;
    #         # it is better to have few calls producing big arrays
    #         array, sig, eps = self.compute(gs, num_events, mean_stds[g])
    #         M, N, E = array.shape
    #         for n in range(N):
    #             for e in range(E):
    #                 if (array[:, n, e] < min_iml).all():
    #                     array[:, n, e] = 0
    #         array = array.transpose(1, 0, 2)  # from M, N, E to N, M, E
    #         n = 0
    #         for rlz in rlzs:
    #             eids = eids_by_rlz[rlz]
    #             for ei, eid in enumerate(eids):
    #                 gmfa = array[:, :, n + ei]  # shape (N, M)
    #                 if sig_eps is not None:
    #                     tup = tuple([eid, rlz] + list(sig[:, n + ei]) +
    #                                 list(eps[:, n + ei]))
    #                     sig_eps.append(tup)
    #                 items = []
    #                 for sp in self.sec_perils:
    #                     o = sp.compute(mag, zip(self.imts, gmfa.T), self.ctx)
    #                     for outkey, outarr in zip(sp.outputs, o):
    #                         items.append((outkey, outarr))
    #                 for i, gmv in enumerate(gmfa):
    #                     if gmv.sum() == 0:
    #                         continue
    #                     data['sid'].append(sids[i])
    #                     data['eid'].append(eid)
    #                     data['rlz'].append(rlz)  # used in compute_gmfs_curves
    #                     for m in range(M):
    #                         data[f'gmv_{m}'].append(gmv[m])
    #                     for outkey, outarr in items:
    #                         data[outkey].append(outarr[i])
    #                     # gmv can be zero due to the minimum_intensity, coming
    #                     # from the job.ini or from the vulnerability functions
    #             n += len(eids)
    #     return data, time.time() - t0

    def compute(self, gsim, num_events, mean_stds, norm_res=None):
        """
        :param gsim: GSIM used to compute mean_stds
        :param num_events: the number of seismic events
        :param mean_stds: array of shape O, M, N
        :returns:
            a 32 bit array of shape (num_imts, num_sites, num_events) and
            two arrays with shape (num_imts, num_events): sig for stddev_inter
            and eps for the random part
        """
        result = np.zeros(
                 (len(self.imts), len(self.ctx.sids), num_events), F32)
        if norm_res is not None:
            if len(self.imts) != norm_res.shape[1]:
                raise Exception("number of imt in norm_res does not correspond")
            if len(self.ctx.sids) != norm_res.shape[0]:
                raise Exception("number of sites in norm_res does not correspond")
            #TODO
            # if num_events != norm_res.shape[2]:
            #     raise Exception("number of events (i.e., simulations) in norm_res does not correspond")
        
        for imti, imt in enumerate(self.imts):
            if norm_res is None:
                result[imti] = self._compute(
                  mean_stds[:, :, imti], num_events, imt, gsim, None)
            else:
                result[imti] = self._compute(
                  mean_stds[:, :, imti], num_events, imt, gsim, norm_res[:, imti])
        return result


    def _compute(self, mean_stds, num_events, imt, gsim, norm_res=None):
        """
        :param mean_stds: array of shape (O, N)
        :param num_events: the number of seismic events
        :param imt: an IMT instance
        :param gsim: a GSIM instance
        :returns: (gmf(num_sites, num_events), stddev_inter(num_events),
                    epsilons(num_events))
        """
        # num_sids = len(self.sids)
        # num_outs = len(mean_stds)
        num_sids = len(self.ctx.sids)
        if self.distribution is None:
            # for truncation_level = 0 there is only mean, no stds
            if self.correlation_model:
                raise ValueError('truncation_level=0 requires '
                                 'no correlation model')
            mean = mean_stds[0]
            gmf = to_imt_unit_values(mean, imt)
            gmf = gmf.T
            gmf = gmf.repeat(num_events, axis=1)
            return gmf
        # elif gsim.DEFINED_FOR_STANDARD_DEVIATION_TYPES == {StdDev.TOTAL}:
        #     raise Exception("not done")
        #     # If the GSIM provides only total standard deviation, we need
        #     # to compute mean and total standard deviation at the sites
        #     # of interest.
        #     # In this case, we also assume no correlation model is used.
        #     if self.correlation_model:
        #         raise CorrelationButNoInterIntraStdDevs(
        #             self.correlation_model, gsim)

        #     mean, stddev_total = mean_stds[:2]
        #     stddev_total = stddev_total.reshape(stddev_total.shape + (1, ))
        #     mean = mean.reshape(mean.shape + (1, ))

        #     total_residual = stddev_total * rvs(
        #         self.distribution, num_sids, num_events)
        #     gmf = to_imt_unit_values(mean + total_residual, imt)
        #     stdi = numpy.nan
        #     epsilons = numpy.empty(num_events, F32)
        #     epsilons.fill(numpy.nan)
        else:
            mean, stddev_total, stddev_inter, stddev_intra = mean_stds
            stddev_intra = stddev_intra.T
            stddev_inter = stddev_inter.T
            mean = mean.T
            if norm_res is None:
                intra_residual = stddev_intra * rvs(
                    self.distribution, num_sids, num_events)
            else:
                if len(norm_res.shape) == 1:
                    norm_res.shape += (1,)
                intra_residual = stddev_intra * norm_res

            epsilons = rvs(self.distribution, num_events)
            inter_residual = stddev_inter * epsilons

            gmf = to_imt_unit_values(
                mean + intra_residual + inter_residual, imt)
        return gmf

    
    def set_seed(self, seed):
        np.random.seed(seed)
        


#%% small test for GmfComputerMod

if __name__ == "__main__":
    
    from scipy.stats.mstats import gmean
    import matplotlib.pyplot as plt
    from pysimulator.rupture_builder import RuptureBuilder
    from pyplotting.map_earthquakes import MapEarthquakes
    from pyerf.distributed.modify_distributed_mfds import get_polygon_influence
    from openquake.hazardlib.calc.gmf import GmfComputer
    
    # general settings
    grid = 0.1
    
    # rupture settings
    mag = 6.0
    strike = 0.
    dip = 60
    rake = 0.
    lon, lat, depth = 0., 0., 5.

    # gmpe settings
    params = {
              'calculation_mode': 'event_based',
              "gsim": "CauzziEtAl2014", # CauzziEtAl2014 # BindiEtAl2011 # BindiEtAl2014Rhyp # Bradley2013
              "reference_backarc": False,
              'reference_vs30_type': 'measured',
              'reference_vs30_value': '800.0',
              'reference_depth_to_2pt5km_per_sec': '1.0',
              'reference_depth_to_1pt0km_per_sec': '19.367',
              'truncation_level': '3',
              'maximum_distance': '200.0',
              }
    
    # grid settings
    lons_bins = np.arange(-1., 1., grid)+grid/2
    lats_bins = np.arange(-1., 1., grid)+grid/2
    lons, lats = np.meshgrid(lons_bins, lats_bins)
    lons = lons.flatten()
    lats = lats.flatten()

    
    #%% some checks on median ground motion and std with magnitude
    
    T_sim=np.array([0.01, 0.1, 0.2, 0.5, 1.])
    cols = ['r','b','g', 'm', [1, 0.5, 0], [0.5, 1, 0]]
    mags = np.arange(5,8,0.1)
    

    # mean + total std only
    fig, axs = plt.subplots(3, 1, figsize=(7.,16), sharex=True)

    for dist, ax in zip([0., 0.2, 1.], axs):
        mean_gms_m = list()
        mean_gms = list()
        mean_gms_p = list()
        gm = GroundMotion(params, [dist], [0.], T_sim=T_sim)
        for mag in mags:    
            rup = RuptureBuilder.init_surface_from_point(mag, lon, lat, depth,
                                                         strike, dip, rake)
            gmf_computer = GmfComputerMod(rupture=rup,
                                          sitecol=gm.sitecol, 
                                          cmaker=gm.cmaker)
            mean_stds = gm.cmaker.get_mean_stds([gmf_computer.ctx])
            m_m = mean_stds[0,0,:,0] - mean_stds[1,0,:,0]
            m = mean_stds[0,0,:,0]
            m_p = mean_stds[0,0,:,0] + mean_stds[1,0,:,0]

            mean_gms_m.append(to_imt_unit_values(m_m, ""))
            mean_gms.append(to_imt_unit_values(m, ""))
            mean_gms_p.append(to_imt_unit_values(m_p, ""))
        mean_gms_m = np.array(mean_gms_m)
        mean_gms = np.array(mean_gms)
        mean_gms_p = np.array(mean_gms_p)
        
        for t, ts in enumerate(T_sim):
            ax.plot(mags, mean_gms[:,t], color=cols[t],
                    label="SA("+str(ts)+"s)")
            # ax.fill_between(mags, mean_gms_m[:,t], mean_gms_p[:,t],
            #                 color=cols[t], alpha=.2)
        ax.set_ylabel("Medians, distance: "+str(dist)+" deg")

    ax.set_xlabel("Magnitude")
    ax.legend()
    plt.show()
    


    #%% mean stds with periods
    
    fig, ax = plt.subplots(1,1, figsize=(7.,6.))
    gm = GroundMotion(params, [0.], [0.])
    for i, mag in enumerate(np.arange(5, 8., 0.5)):
        rup = RuptureBuilder.init_surface_from_point(mag, lon, lat, depth,
                                                     strike, dip, rake)
        gmf_computer = GmfComputerMod(rupture=rup,
                                      sitecol=gm.sitecol, 
                                      cmaker=gm.cmaker)
        mean_stds = gm.cmaker.get_mean_stds([gmf_computer.ctx])
        ax.plot(gm.T_sim, to_imt_unit_values(mean_stds[0,0,:,0],""),
                color=cols[i], label="mag "+str(mag))
    ax.set_ylabel("mean")
    ax.set_xlabel("T(s)")
    ax.set_xscale("log")
    ax.legend()
    plt.show()
    

    
    fig, ax = plt.subplots(1,1, figsize=(7.,6.))
    gm = GroundMotion(params, [0.], [0.])
    rup = RuptureBuilder.init_surface_from_point(mag, lon, lat, depth,
                                                 strike, dip, rake)
    gmf_computer = GmfComputerMod(rupture=rup,
                                  sitecol=gm.sitecol, 
                                  cmaker=gm.cmaker)
    mean_stds = gm.cmaker.get_mean_stds([gmf_computer.ctx])
    legs = ["stddev total", "stddev inter", "stddev intra"]
    for i in range(1,4):
        ax.plot(gm.T_sim, mean_stds[i,0,:,0], color=cols[i-1], label=legs[i-1])
    ax.set_ylabel("std")
    ax.set_xlabel("T(s)")
    ax.set_xscale("log")
    ax.legend()
    plt.show()
    



    #%% plot median and median + std fixing distance and varying mag for SA(0.1)
    


    mags = np.arange(5, 8., 0.1)
    
    params["gsim"] = "Bradley2013"
    gm = GroundMotion(params, [0.], [0.], np.array([0.01, 0.2, 0.5, 1.0]))
    mediansB = dict()
    stdsB = dict()
    for i, mag in enumerate(mags):
        rup = RuptureBuilder.init_surface_from_point(mag, lon, lat, depth,
                                                     strike, dip, rake)
        gmf_computer = GmfComputerMod(rupture=rup,
                                      sitecol=gm.sitecol, 
                                      cmaker=gm.cmaker)
        mean_stds = gm.cmaker.get_mean_stds([gmf_computer.ctx])
        for j, SA in enumerate(gm.T_sim):
            if SA not in mediansB.keys():
                mediansB[SA] = list()
            if SA not in stdsB.keys():
                stdsB[SA] = list()
            mediansB[SA].append(mean_stds[0,0,j,0])
            stdsB[SA].append(mean_stds[1,0,j,0])
    

    params["gsim"] = "CauzziEtAl2014" # CauzziEtAl2014 # BindiEtAl2011 # BindiEtAl2014Rhyp # Bradley2013
    gm = GroundMotion(params, [0.], [0.], np.array([0.01, 0.2, 0.5, 1.0]))
    mediansC = dict()
    stdsC = dict()
    for i, mag in enumerate(mags):
        rup = RuptureBuilder.init_surface_from_point(mag, lon, lat, depth,
                                                     strike, dip, rake)
        gmf_computer = GmfComputerMod(rupture=rup,
                                      sitecol=gm.sitecol, 
                                      cmaker=gm.cmaker)
        mean_stds = gm.cmaker.get_mean_stds([gmf_computer.ctx])
        for j, SA in enumerate(gm.T_sim):
            if SA not in mediansC.keys():
                mediansC[SA] = list()
            if SA not in stdsC.keys():
                stdsC[SA] = list()
            mediansC[SA].append(mean_stds[0,0,j,0])
            stdsC[SA].append(mean_stds[1,0,j,0])

    
    from scipy.stats import norm, truncnorm
    x_axis = np.arange(-7.5, 5, 0.001)

    ind1 = np.where(np.isclose(mags,5.4))[0][0]
    ind2 = np.where(np.isclose(mags,6.7))[0][0]

    fig, ax = plt.subplots(1,1, figsize=(7.,6.))
    plt.plot(x_axis, truncnorm.pdf(x_axis,-3,3, mediansB[1.][ind1],stdsB[1.][ind1]), color="r",
             linestyle="--", label="Bradley SA(1.0) M=5.4")
    plt.plot(x_axis, truncnorm.pdf(x_axis,-3,3, mediansB[1.][ind2],stdsB[1.][ind2]), color="r",
             linestyle="--", label="Bradley SA(1.0) M=6.7")
    plt.xlabel("log(IM)")
    plt.ylabel("PDF")
    plt.legend()
    # plt.xlim([0,4])
    plt.ylim([0.,0.7])
    plt.show()


    fig, ax = plt.subplots(1,1, figsize=(7.,6.))
    plt.plot(x_axis, truncnorm.pdf(x_axis,-3,3, mediansC[1.][ind1],stdsC[1.][ind1]), color="b",
             label="Cauzzi SA(1.0) M=5.4")
    plt.plot(x_axis, truncnorm.pdf(x_axis,-3,3, mediansC[1.][ind2],stdsC[1.][ind2]), color="b",
             linestyle="--", label="Cauzzi SA(1.0) M=6.7")
    plt.xlabel("log(IM)")
    plt.ylabel("PDF")
    plt.legend()
    # plt.xlim([0,4])
    plt.ylim([0,0.7])
    plt.show()


    from scipy.integrate import simps

    probB = list()
    probC = list()
    for j, SA in enumerate(gm.T_sim):
        y = truncnorm.pdf(x_axis,-3,3, mediansB[SA][ind2],stdsB[SA][ind2]) * \
                       (1.-norm.cdf(x_axis,mediansB[SA][ind1],stdsB[SA][ind1]))
        probB.append(simps(y, x_axis))
        y = truncnorm.pdf(x_axis,-3,3, mediansC[SA][ind2],stdsC[SA][ind2]) * \
                       (1.-norm.cdf(x_axis,mediansC[SA][ind1],stdsC[SA][ind1]))
        probC.append(simps(y, x_axis))


    fig, axs = plt.subplots(2,1, sharex=True, figsize=(7.,6.))
    axs[0].plot(gm.T_sim, probB, color="r", label="Bradley")
    axs[0].plot(gm.T_sim, probC, color="b", label="Cauzzi")
    axs[0].set_ylabel("Probability gm higher")
    axs[0].legend()
    
    axs[1].plot(gm.T_sim, np.array(probB)/probB[0], 
                color="r", label="Bradley")
    axs[1].plot(gm.T_sim, np.array(probC)/probC[0],
                color="b", label="Cauzzi")
    axs[1].set_xlabel("Structural period (s)")
    axs[1].set_ylabel("Relative probability decrease wrt peak")
    # axs[1].set_xlim(0,1.)
    # axs[1].set_xscale("log")
    plt.show()

    s






    #%% inputs for the rest

    # rupture
    rup = RuptureBuilder.init_surface_from_point(mag, lon, lat, depth,
                                                 strike, dip, rake)
    
    # context maker
    gm = GroundMotion(params, lons, lats)
    
    # gm = GroundMotion(params, lons, lats, np.array([0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
    #                     0.3,  0.4,  0.5, 0.75, 1.   , 1.5, 2.  , 3. , 4.])) # BindiEtAl2011
    
    # gm = GroundMotion(params, lons, lats, np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
    #                     0.3,  0.4,  0.5, 0.75, 1.   , 1.5, 2.  , 3.])) # BindiEtAl2014Rhyp
      

    #%% mean only
    
    np.random.seed(42)
    num_simulations = 1
    
    gmf_computer = GmfComputerMod(rupture=rup,
                                  sitecol=gm.sitecol, 
                                  cmaker=gm.cmaker)
    mean_stds = gm.cmaker.get_mean_stds([gmf_computer.ctx])
    gmf_computer = GmfComputerMod(rupture=rup,
                                  sitecol=gm.sitecol, 
                                  cmaker=gm.cmaker)
    mean_stds = gm.cmaker.get_mean_stds([gmf_computer.ctx])
    gmf_computer.distribution = None
    gmf_m = gmf_computer.compute(params["gsim"], num_simulations, mean_stds)

    # map
    xx = lons.reshape((lons_bins.shape[0], lats_bins.shape[0]))
    yy = lats.reshape((lons_bins.shape[0], lats_bins.shape[0]))
    zz = gmf_m[0,:,0].reshape((lons_bins.shape[0], lats_bins.shape[0]))
  
    pols = get_polygon_influence(xx.flatten(), yy.flatten())

    mpe = MapEarthquakes()
    mpe.extent = [-1,1,-1,1]
    con = mpe.add_colored_polygons(pols, zz.flatten(), cmap="jet",
                                   linewidth=0.0)
    cb = mpe.add_colorbar(con, location="right",
                          label="_st")
    # con = mpe.ax.contourf(xx, yy, zz, alpha=0.5, cmap="jet")
    # mpe.ax.scatter(lons, lats, s=2, color=[0.5,0.5,0.5], alpha=0.5)
    mpe.ax.scatter(rup.surface.mesh.lons.flatten(),
                    rup.surface.mesh.lats.flatten(),
                    s=2, color="r", alpha=0.5)
    # cb = mpe.fig.colorbar(con, orientation="horizontal", label="PGA")
    mpe.look(lon_step = 0.2, lat_step = 0.2)
    mpe.show()
        
    
    # plot response spectrum for max pga index
    ind = np.argmax(gmf_m[0,:,0])
    rs_m = gmf_m[:,ind,0]
    plt.figure()
    plt.plot(gm.T_sim,rs_m)
    plt.show()
    
    


    #%% no pca
    
    np.random.seed(42)
    num_simulations = 2 # of the same event
    
    gmf_computer = GmfComputerMod(rupture=rup,
                                  sitecol=gm.sitecol, 
                                  cmaker=gm.cmaker)
    mean_stds = gm.cmaker.get_mean_stds([gmf_computer.ctx])
    gmf = gmf_computer.compute(params["gsim"], num_simulations, mean_stds)
    
    # check that over 10000 simulations, the mean is equal to the mean of the GMPE
    # np.testing.assert_allclose(gmf_m[:,:,0], scipy.stats.mstats.gmean(gmf, axis=2))

    # map
    xx = lons.reshape((lons_bins.shape[0], lats_bins.shape[0]))
    yy = lats.reshape((lons_bins.shape[0], lats_bins.shape[0]))
    zz = gmf[0,:,0].reshape((lons_bins.shape[0], lats_bins.shape[0]))
    mpe = MapEarthquakes([-1,1,-1,1])
    mpe.lon_step = 0.2
    mpe.lat_step = 0.2
    con = mpe.ax.contourf(xx, yy, zz, alpha=0.5, cmap="jet")
    # mpe.ax.scatter(lons, lats, s=2, color=[0.5,0.5,0.5], alpha=0.5)
    mpe.ax.scatter(rup.surface.mesh.lons.flatten(),
                    rup.surface.mesh.lats.flatten(),
                    s=2, color="r", alpha=0.5)
    cb = mpe.fig.colorbar(con, orientation="horizontal", label="PGA")
    mpe.look()
    mpe.show()
        
    # plot response spectrum for max pga index
    rs = gmf[:,ind,:]
    plt.figure()
    plt.plot(gm.T_sim, rs, color=[0.5,0.5,0.5])
    plt.plot(gm.T_sim, rs_m, color="r")
    plt.plot(gm.T_sim, gmean(gmf, axis=2)[:,ind])
    plt.show()

    
    #%% with pca

    np.random.seed(42)
    num_simulations = 2 # of the same event
    
    gmf_computer = GmfComputerMod(rupture=rup,
                                  sitecol=gm.sitecol, 
                                  cmaker=gm.cmaker)
    mean_stds = gm.cmaker.get_mean_stds([gmf_computer.ctx])
    norm_res = gm.get_norm_res(num_simulations)
    
    ################### this is to apply truncation (#TODO find a better way!)
    rep = [np.any(norm_res[:,:,i]>gm.oq.truncation_level) or
           np.any(norm_res[:,:,i]<-gm.oq.truncation_level) 
           for i in range(0, num_simulations)]
    while np.any(rep):
        norm_res2 = gm.get_norm_res(np.sum(rep))
        norm_res[:,:,rep] = norm_res2
        rep = [np.any(norm_res[:,:,i]>gm.oq.truncation_level) or
               np.any(norm_res[:,:,i]<-gm.oq.truncation_level) 
               for i in range(0, num_simulations)]
    ############################################## this is to apply truncation
    
    gmf_pca = gmf_computer.compute(params["gsim"], num_simulations, mean_stds, norm_res)
    
    # check that over 10000 simulations, the mean is equal to the mean of the GMPE
    # np.testing.assert_allclose(gmf_m[:,:,0], scipy.stats.mstats.gmean(gmf, axis=2))

    # map
    xx = lons.reshape((lons_bins.shape[0], lats_bins.shape[0]))
    yy = lats.reshape((lons_bins.shape[0], lats_bins.shape[0]))
    zz = gmf_pca[0,:,0].reshape((lons_bins.shape[0], lats_bins.shape[0]))
    mpe = MapEarthquakes([-1,1,-1,1])
    mpe.lon_step = 0.2
    mpe.lat_step = 0.2
    con = mpe.ax.contourf(xx, yy, zz, alpha=0.5, cmap="jet")
    # mpe.ax.scatter(lons, lats, s=2, color=[0.5,0.5,0.5], alpha=0.5)
    mpe.ax.scatter(rup.surface.mesh.lons.flatten(),
                   rup.surface.mesh.lats.flatten(),
                   s=2, color="r", alpha=0.5)
    cb = mpe.fig.colorbar(con, orientation="horizontal", label="PGA")
    mpe.look()
    mpe.show()
        
    # plot response spectrum for max pga index
    rs = gmf_pca[:,ind,:]
    plt.figure()
    plt.plot(gm.T_sim, rs, color=[0.5,0.5,0.5])
    plt.plot(gm.T_sim, rs_m, color="r")
    plt.plot(gm.T_sim, gmean(gmf_pca, axis=2)[:,ind])
    plt.show()


    
