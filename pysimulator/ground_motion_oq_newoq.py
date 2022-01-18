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

from openquake.baselib.general import AccumDict
from openquake.hazardlib.const import StdDev
from openquake.hazardlib.gsim.base import ContextMaker
from openquake.hazardlib.imt import from_string
from openquake.hazardlib.calc.gmf import exp, rvs
from openquake.hazardlib.site import SiteCollection
from openquake.hazardlib.gsim.base import ContextMaker, FarAwayRupture
from openquake.hazardlib.cross_correlation import NoCrossCorrelation
from openquake.commonlib import oqvalidation
from openquake.calculators import base
from openquake.hazardlib import valid

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
    
    def __init__(self, params, lons, lats, T_sim=None):
        
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
        # (4, G, M, N)
        mean_stds = cmaker.get_mean_stds([gmf_computer.ctx for gmf_computer in computers])
        c = 0 # keep track of events (to get the normalized residuals)
        gmfs = list()
        datetimes = list()
        mainshock_flags = list()
        for r, rup in enumerate(catalog.catalog["rupture"]):
            gmf_computer = computers[r]
            if spatialpca:
                res = gmf_computer.compute(gsims[0], 1,
                                           mean_stds[:,0,:,r*num_sites:(r+1)*num_sites],
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
    # The GmfComputer is called from the OpenQuake Engine. In that case
    # the rupture is an higher level containing a
    # :class:`openquake.hazardlib.source.rupture.Rupture` instance as an
    # attribute. Then the `.compute(gsim, num_events, ms)` method is called and
    # a matrix of size (I, N, E) is returned, where I is the number of
    # IMTs, N the number of affected sites and E the number of events. The
    def __init__(self, rupture, sitecol, cmaker, correlation_model=None,
                 cross_correl=None, amplifier=None, sec_perils=()):
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
        self.cross_correl = cross_correl or NoCrossCorrelation(
            cmaker.trunclevel)

    # def compute_all(self, sig_eps=None):
    #     """
    #     :returns: (dict with fields eid, sid, gmv_X, ...), dt
    #     """
    #     min_iml = self.cmaker.min_iml
    #     rlzs_by_gsim = self.cmaker.gsims
    #     t0 = time.time()
    #     sids = self.ctx.sids
    #     eids_by_rlz = self.ebrupture.get_eids_by_rlz(rlzs_by_gsim)
    #     mag = self.ebrupture.rupture.mag
    #     data = AccumDict(accum=[])
    #     mean_stds = self.cmaker.get_mean_stds([self.ctx])  # (4, G, M, N)
    #     for g, (gs, rlzs) in enumerate(rlzs_by_gsim.items()):
    #         num_events = sum(len(eids_by_rlz[rlz]) for rlz in rlzs)
    #         if num_events == 0:  # it may happen
    #             continue
    #         # NB: the trick for performance is to keep the call to
    #         # .compute outside of the loop over the realizations;
    #         # it is better to have few calls producing big arrays
    #         array, sig, eps = self.compute(gs, num_events, mean_stds[:, g])
    #         M, N, E = array.shape  # sig and eps have shapes (M, E) instead
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
        :param mean_stds: array of shape (4, M, N)
        :returns:
            a 32 bit array of shape (num_imts, num_sites, num_events) and
            two arrays with shape (num_imts, num_events): sig for tau
            and eps for the random part
        """
        if norm_res is not None:
            if len(self.imts) != norm_res.shape[1]:
                raise Exception("number of imt in norm_res does not correspond")
            if len(self.ctx.sids) != norm_res.shape[0]:
                raise Exception("number of sites in norm_res does not correspond")
            #TODO
            # if num_events != norm_res.shape[2]:
            #     raise Exception("number of events (i.e., simulations) in norm_res does not correspond")
        M = len(self.imts)
        result = np.zeros(
            (len(self.imts), len(self.ctx.sids), num_events), F32)
        sig = np.zeros((M, num_events), F32)  # same for all events
        eps = np.zeros((M, num_events), F32)  # not the same
        # numpy.random.seed(self.seed)
        num_sids = len(self.ctx.sids)
        if self.cross_correl.distribution:
            # build arrays of random numbers of shape (M, N, E) and (M, E)
            intra_eps = [
                rvs(self.cross_correl.distribution, num_sids, num_events)
                for _ in range(M)]
            inter_eps = self.cross_correl.get_inter_eps(self.imts, num_events)
        else:
            intra_eps = [None] * M
            inter_eps = [np.zeros(num_events)] * M
        for m, imt in enumerate(self.imts):
            try:
                if norm_res is None:
                    result[m], sig[m], eps[m] = self._compute(
                        mean_stds[:, m], imt, gsim, intra_eps[m], inter_eps[m],
                        None)
                else:
                    result[m], sig[m], eps[m] = self._compute(
                        mean_stds[:, m], imt, gsim, intra_eps[m], inter_eps[m],
                        norm_res[:, m])
            except Exception as exc:
                raise RuntimeError(
                    '(%s, %s) %s: %s' %
                    (gsim, imt, exc.__class__.__name__, exc)
                    ).with_traceback(exc.__traceback__)
        if self.amplifier:
            self.amplifier.amplify_gmfs(
                self.ctx.ampcode, result, self.imts, self.seed)
        return result, sig, eps

    def _compute(self, mean_stds, imt, gsim, intra_eps, inter_eps):
        if self.cmaker.trunclevel == 0:
            # for truncation_level = 0 there is only mean, no stds
            if self.correlation_model:
                raise ValueError('truncation_level=0 requires '
                                 'no correlation model')
            mean, _, _, _ = mean_stds
            gmf = exp(mean, imt)[:, None]
            gmf = gmf.repeat(len(inter_eps), axis=1)
            inter_sig = 0
        elif gsim.DEFINED_FOR_STANDARD_DEVIATION_TYPES == {StdDev.TOTAL}:
            raise Exception("to complete")
            # # If the GSIM provides only total standard deviation, we need
            # # to compute mean and total standard deviation at the sites
            # # of interest.
            # # In this case, we also assume no correlation model is used.
            # if self.correlation_model:
            #     raise CorrelationButNoInterIntraStdDevs(
            #         self.correlation_model, gsim)

            # mean, sig, _, _ = mean_stds
            # gmf = exp(mean[:, None] + sig[:, None] * intra_eps, imt)
            # inter_sig = np.nan
        else:
            mean, sig, tau, phi = mean_stds
            # the [:, None] is used to implement multiplication by row;
            # for instance if  a = [1 2], b = [[1 2] [3 4]] then
            # a[:, None] * b = [[1 2] [6 8]] which is the expected result;
            # otherwise one would get multiplication by column [[1 4] [3 8]]
            intra_res = phi[:, None] * intra_eps  # shape (N, E)

            if self.correlation_model is not None:
                intra_res = self.correlation_model.apply_correlation(
                    self.sites, imt, intra_res, phi)
                if len(intra_res.shape) == 1:  # a vector
                    intra_res = intra_res[:, None]

            inter_res = tau[:, None] * inter_eps  # shape (N, 1) * E => (N, E)
            gmf = exp(mean[:, None] + intra_res + inter_res, imt)  # (N, E)
            inter_sig = tau.max()  # from shape (N, 1) => scalar
        return gmf, inter_sig, inter_eps  # shapes (N, E), 1, E

    
    def set_seed(self, seed):
        np.random.seed(seed)
        
    
    
#%% small test for GmfComputerMod

if __name__ == "__main__":
    
    from pysimulator.rupture_builder import RuptureBuilder
    from pyetas.map_earthquakes import MapEarthquakes
    from openquake.hazardlib.calc.gmf import GmfComputer
    
    # rupture settings
    mag = 6.0
    strike = 0.
    dip = 60
    rake = 0.
    lon, lat, depth = 0., 0., 0.

    # gmpe settings
    params = {
              'calculation_mode': 'event_based',
              "gsim": "BooreAtkinson2011",
              'reference_vs30_value': '800.0',
              'truncation_level': '3',
              'maximum_distance': '200.0',
              }
    
    # grid settings
    lons_bins = np.arange(-1., 1.1, 0.05)
    lats_bins = np.arange(-1., 1.1, 0.05)
    lons, lats = np.meshgrid(lons_bins, lats_bins)
    lons = lons.flatten()
    lats = lats.flatten()

    # rupture
    rup = RuptureBuilder.init_surface_from_point(mag, lon, lat, depth,
                                                 strike, dip, rake)
    
    # context maker
    gm = GroundMotion(params, lons, lats)
    
    
    #%% compute gmf
    
    gmf_computer = GmfComputerMod(rupture=rup,
                                  sitecol=gm.sitecol, 
                                  cmaker=gm.cmaker)
    
    mean_stds = gm.cmaker.get_mean_stds([gmf_computer.ctx])
    gmf_computer.compute(params["gsim"], 1, mean_stds)
    
    
    #%%

    xx = lons.reshape((lons_bins.shape[0], lats_bins.shape[0]))
    yy = lats.reshape((lons_bins.shape[0], lats_bins.shape[0]))
    zz = rate5.reshape((lons_bins.shape[0], lats_bins.shape[0]))

    
    mpe = MapEarthquakes([-1,1,-1,1])
    con = mpe.ax.contourf(xx, yy, zz, alpha=0.5, cmap="jet")
    mpe.ax.scatter(lons, lats, s=2, color=[0.5,0.5,0.5], alpha=0.5)
    mpe.ax.scatter(rup.surface.mesh.lons.flatten(),
                   rup.surface.mesh.lats.flatten(),
                   s=2, color="r", alpha=0.5)
    cb = mpe.ax.colorbar(con, location='right', label="PGA")
    mpe.look()
    mpe.show()
        
        
