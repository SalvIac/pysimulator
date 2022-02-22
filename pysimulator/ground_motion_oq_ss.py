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

import time
import numpy as np
import scipy.stats
from tqdm import tqdm

from openquake.baselib.general import AccumDict
from openquake.hazardlib.const import StdDev
from openquake.hazardlib.gsim.base import ContextMaker
from openquake.hazardlib.imt import from_string
from openquake.hazardlib.calc.gmf import to_imt_unit_values, rvs
from openquake.hazardlib.site import SiteCollection
from openquake.hazardlib.gsim.base import ContextMaker
from openquake.commonlib import oqvalidation
from openquake.calculators import base
from openquake.hazardlib import valid

from pyrisk.spatial_pca.spatial_pca_residuals import SpatialPcaResiduals
from pysimulator.ground_motion_container import (GmfGroupContainer,
                                                      GmfCatalogContainer)
from pysimulator.run_multiprocess import run_multiprocess

F32 = np.float32


class GroundMotion():
    
    multiprocessing = False
    cores = 4
    seed = 42
    
    def __init__(self, params, lons, lats, T_sim):
        
        self.lons = lons
        self.lats = lats
        self.T_sim = T_sim
        params['intensity_measure_types'] = self.get_imts(T_sim) #'PGA, SA(0.02), SA(0.03), SA(0.05), SA(0.075), SA(0.1), SA(0.15), SA(0.2), SA(0.25), SA(0.3), SA(0.4), SA(0.5), SA(0.75), SA(1.0), SA(1.5), SA(2.0), SA(3.0), SA(4.0), SA(5.0)',
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
        
        # context maker
        self.gsims = [valid.gsim(self.oq.gsim)]
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
            nsims = 0
            for catalog in catalogs:
                nsims += catalog.get_num_events()
            print("number of normalized residuals: "+str(nsims))
            # this is for a faster simulations
            norm_res = self.get_norm_res(nsims)
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
        
        c = 0 # keep track of events (to get the normalized residuals)
        gmfs = list()
        datetimes = list()
        mainshock_flags = list()
        for r, rup in enumerate(catalog.catalog["rupture"]):
            gmf_computer = GmfComputerMod(rupture=rup,
                                          sitecol=sitecol, 
                                          cmaker=cmaker)
            [mean_stds] = cmaker.get_mean_stds([gmf_computer.ctx],
                                                StdDev.EVENT)
            if spatialpca:
                res = gmf_computer.compute(gsims[0], 1, mean_stds,
                                           norm_res[:,:,c])
            else:
                res = gmf_computer.compute(gsims[0], 1, mean_stds)
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
    
    spr = None
    
    # The GmfComputer is called from the OpenQuake Engine. In that case
    # the rupture is an higher level containing a
    # :class:`openquake.hazardlib.source.rupture.Rupture` instance as an
    # attribute. Then the `.compute(gsim, num_events, ms)` method is called and
    # a matrix of size (I, N, E) is returned, where I is the number of
    # IMTs, N the number of affected sites and E the number of events. The
    def __init__(self, rupture, sitecol, cmaker):
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
        self.source_id = '?'
        self.ctx, sites, dctx = cmaker.make_contexts(sitecol, rupture)
        vars(self.ctx).update(vars(dctx))
        for par in sites.array.dtype.names:
            setattr(self.ctx, par, sites[par])
        self.sids = sites.sids
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
        if norm_res is not None:
            if len(self.imts) != norm_res.shape[1]:
                raise Exception("number of imt in norm_res does not correspond")
            if len(self.sids) != norm_res.shape[0]:
                raise Exception("number of sites in norm_res does not correspond")
            #TODO
            # if num_events != norm_res.shape[2]:
            #     raise Exception("number of events (i.e., simulations) in norm_res does not correspond")
        
        result = np.zeros((len(self.imts), len(self.sids), num_events), F32)
        
        for imti, imt in enumerate(self.imts):
            if norm_res is None:
                result[imti] = self._compute(
                        mean_stds[:, imti], num_events, imt, gsim, None)
            else:
                result[imti] = self._compute(
                        mean_stds[:, imti], num_events, imt, gsim, norm_res[:, imti])
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
        num_sids = len(self.sids)
        num_outs = len(mean_stds)
        # if num_outs == 1:
        #     # for truncation_level = 0 there is only mean, no stds
        #     mean = mean_stds[0]
        #     gmf = to_imt_unit_values(mean, imt)
        #     gmf.shape += (1, )
        #     gmf = gmf.repeat(num_events, axis=1)
        #     return (gmf,
        #             np.zeros(num_events, F32),
        #             np.zeros(num_events, F32))
        # elif num_outs == 2:
        #     # If the GSIM provides only total standard deviation, we need
        #     # to compute mean and total standard deviation at the sites
        #     # of interest.
        #     # In this case, we also assume no correlation model is used.
        #     mean, stddev_total = mean_stds
        #     stddev_total = stddev_total.reshape(stddev_total.shape + (1, ))
        #     mean = mean.reshape(mean.shape + (1, ))

        #     total_residual = stddev_total * rvs(
        #         self.distribution, num_sids, num_events)
        #     gmf = to_imt_unit_values(mean + total_residual, imt)
        #     stdi = np.nan
        #     epsilons = np.empty(num_events, F32)
        #     epsilons.fill(np.nan)
        if num_outs == 3:
            mean, stddev_inter, stddev_intra = mean_stds
            stddev_intra = stddev_intra.reshape(stddev_intra.shape + (1, ))
            stddev_inter = stddev_inter.reshape(stddev_inter.shape + (1, ))
            mean = mean.reshape(mean.shape + (1, ))
            if norm_res is None:
                intra_residual = stddev_intra * rvs(
                    self.distribution, num_sids, num_events)
            else:
                intra_residual = stddev_intra * norm_res.reshape(norm_res.shape + (1, ))

            epsilons = rvs(self.distribution, num_events)
            inter_residual = stddev_inter * epsilons
            gmf = to_imt_unit_values(mean + intra_residual + inter_residual,
                                     imt)
        return gmf

    
    def set_seed(self, seed):
        np.random.seed(seed)
        
