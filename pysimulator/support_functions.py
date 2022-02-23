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

import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
plt.ioff()
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.pmf import PMF
from pysimulator.simulation_functions import *



global cols
cols = ["b", "r", "g", "m", "c", [1.,0.6,0.], [0.,0.6,1.]]




def string2float(string):
    if string == "" or string == "None":
        return None
    else:
        return float(string)
    

def plot_2d_generic(catalog_to_start=None, exposure=None,
                    include_basemap_flag=0, ax=None, show=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    X, Y = list(), list()
    if exposure is not None:
        for i, ass in enumerate(exposure.assets):
            X.append(ass.location[0])
            Y.append(ass.location[1])
        ax.scatter(X, Y, s=2, color="k", alpha=0.5, linewidth=0.1)    
    
    if catalog_to_start is not None:
        edges = True
        col_id = 0
        for i, rup in enumerate(catalog_to_start.catalog["rupture"]):
            X = rup.surface.mesh.lons
            Y = rup.surface.mesh.lats
            col = cols[col_id]
            ax.scatter(X.flatten(), Y.flatten(), s=2, color=col,
                       alpha=0.5, linewidth=0.1) 
            if edges:
                ax.plot(X[0,:], Y[0,:], color=col, linewidth=0.2, zorder=200)
                ax.plot(X[-1,:], Y[-1,:], color=col, linewidth=0.2, zorder=200)
                ax.plot(X[:,0], Y[:,0], color=col, linewidth=0.2, zorder=200)
                ax.plot(X[:,-1], Y[:,-1], color=col, linewidth=0.2, zorder=200)
            ax.scatter(rup.hypocenter.longitude, rup.hypocenter.latitude,
                       s=50, marker="*", color=col, linewidth=0.1) 
            col_id += 1

    extent = [np.floor(ax.get_xlim()[0]/0.5)*0.5,
              np.ceil(ax.get_xlim()[1]/0.5)*0.5, 
              np.floor(ax.get_ylim()[0]/0.5)*0.5,
              np.ceil(ax.get_ylim()[1]/0.5)*0.5]

    if include_basemap_flag == 1:
        from mpl_toolkits.basemap import Basemap
        bm = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[2],
                      urcrnrlon=extent[1], urcrnrlat=extent[3],
                      projection='cyl', resolution='h', fix_aspect=False, ax=ax)
        ax.add_collection(bm.drawcoastlines(linewidth=0.25))
        ax.add_collection(bm.drawcountries(linewidth=0.35))

    lon_step = 0.5
    lat_step = 0.5
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    meridians = np.round(np.arange(extent[0], extent[1] + lon_step, lon_step), 2)
    parallels = np.round(np.arange(extent[2], extent[3] + lat_step, lat_step), 2)
    ax.set_yticks(parallels)
    ax.set_xticks(meridians)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if show:
        plt.show()
    return ax





def plot_3d_generic(catalog_to_start=None, exposure=None,
                    include_basemap_flag=0, ax=None, show=False):
    from mpl_toolkits.mplot3d import Axes3D
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    
    X, Y, Z = list(), list(), list()
    if exposure is not None:
        for i, ass in enumerate(exposure.assets):
            X.append(ass.location[0])
            Y.append(ass.location[1])
            Z.append(0.)
        ax.scatter3D(X, Y, Z, s=2, color="k", alpha=0.5, linewidth=0.1) 
    
    if catalog_to_start is not None:        
        edges = True
        col_id = 0
        for i, rup in enumerate(catalog_to_start.catalog["rupture"]):
            X = rup.surface.mesh.lons
            Y = rup.surface.mesh.lats
            Z = -rup.surface.mesh.depths
            col = cols[col_id]
            ax.plot_surface(X, Y, Z, color=col, alpha=0.5, linewidth=0.1, 
                                  antialiased=True, edgecolor='k') 
            if edges:
                ax.plot3D(X[0,:], Y[0,:], Z[0,:], color=col, linewidth=0.2, zorder=200)
                ax.plot3D(X[-1,:], Y[-1,:], Z[-1,:], color=col, linewidth=0.2, zorder=200)
                ax.plot3D(X[:,0], Y[:,0], Z[:,0], color=col, linewidth=0.2, zorder=200)
                ax.plot3D(X[:,-1], Y[:,-1], Z[:,-1], color=col, linewidth=0.2, zorder=200)
            col_id += 1
            ax.scatter3D(rup.hypocenter.longitude, rup.hypocenter.latitude,
                         -rup.hypocenter.depth, s=50, marker="*", color=col,
                         linewidth=0.1) 
        # for i, rup in enumerate([rupture]):
        #     X = rup.surface.mesh.lons.flatten()
        #     Y = rup.surface.mesh.lats.flatten()
        #     Z = -rup.surface.mesh.depths.flatten()
        #     ax.scatter3D(X, Y, Z, s=2,  color="k", alpha=0.5, linewidth=0.1) 

    extent = [np.floor(ax.get_xlim()[0]/0.5)*0.5,
              np.ceil(ax.get_xlim()[1]/0.5)*0.5, 
              np.floor(ax.get_ylim()[0]/0.5)*0.5,
              np.ceil(ax.get_ylim()[1]/0.5)*0.5]
    
    print(include_basemap_flag)
    if include_basemap_flag == 1:
        from mpl_toolkits.basemap import Basemap
        bm = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[2],
                      urcrnrlon=extent[1], urcrnrlat=extent[3],
                      projection='cyl', resolution='h', fix_aspect=False, ax=ax)
        ax.add_collection3d(bm.drawcoastlines(linewidth=0.25))
        ax.add_collection3d(bm.drawcountries(linewidth=0.35))

    ax.set_xlabel('Longitude', labelpad=15)
    ax.set_ylabel('Latitude', labelpad=35)
    ax.set_zlabel('Depth (km)', labelpad=10)
    lon_step = 0.5
    lat_step = 0.5
    meridians = np.round(np.arange(extent[0], extent[1] + lon_step, lon_step), 2)
    parallels = np.round(np.arange(extent[2], extent[3] + lat_step, lat_step), 2)
    ax.set_yticks(parallels)
    ax.set_xticks(meridians)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_zlim(-30., 0)
    if show:
        plt.show()
    return ax
    







def plot_depth_pdf():
    print('app_support.plot_depth_pdf')
    sys.stdout.flush()


def plot_mag_freq_distr(b, mmin, mmax):
    m = np.arange(mmin, mmax+0.1, 0.1)
    pdfm = pdf_magnitude_trunc(m, b, mmin, mmax)
    cdfm = cdf_magnitude_trunc(m, b, mmin, mmax)
    fig, ax = plt.subplots(2, 1, sharex=True)
    # pdf
    ax[0].plot(m, pdfm, 'r-', lw=2, alpha=0.6,
                label='truncated GR pdf\nb='+str(b)+'\nm_{min}='+str(mmin)+'\nm_{max}='+str(mmax))
    ax[0].set_ylabel('f(m)')
    ax[0].set_yscale("log")
    ax[0].legend()
    # cdf    
    ax[1].plot(m, cdfm, 'r-', lw=2, alpha=0.6)
    ax[1].set_ylim([0,1])
    ax[1].set_xlabel('m')
    ax[1].set_ylabel('F(m)')
    plt.show()


def plot_mag_incompl(min_mag, args):
    mags = np.arange(min_mag, 9, 0.5)
    args = [1.,4.5,0.75]
    tt = np.arange(0.0001, 10, 0.0001)
    fig = plt.figure()
    for mag in mags:
        mc = compl_vs_time_general(mag, tt, *args)
        plt.plot(tt, mc, label="Mainshock $M_W$ {:.1f}".format(mag))
    plt.xlabel("Time (days)")
    plt.ylabel("Completeness magnitude ($M_W$)")
    plt.legend()
    plt.show()
    

def plot_productivity(A, alpha, mmin):
    print('app_support.plot_productivity')
    sys.stdout.flush()
    mm = np.arange(mmin, 8.5+0.01, 0.01)
    fig = plt.figure()
    num = A*np.exp(alpha*(mm-mmin))
    plt.plot(mm, num, label="Productivity\nA={:.1f}".format(A)+'\nalpha={:.1f}'.format(alpha))
    plt.xlabel("Magnitude ($M_W$)")
    plt.ylabel("Average number of events")
    plt.legend()
    plt.show()


def plot_space_pdf(q, D, gamma, mmin):
    print('app_support.plot_space_pdf')
    sys.stdout.flush()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    dm = 7.-mmin
    dr = 0.02
    xymax = 2 # this is in degree
    x, y = np.mgrid[-xymax:xymax+dr:dr, -xymax:xymax+dr:dr]
    pdfs = pdf_space5(x, y, dm, q, D, gamma)
    cdfs = cdf_space5(x, y, dm, q, D, gamma)
    y0 = int(x.shape[0]/2)
    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(x, y, pdfs, s=0.1, alpha=0.5)
    ax.plot(x[:,y0], y[:,y0], pdfs[:,y0], color='r')
    ax.set_xlabel('x (deg)')
    ax.set_ylabel('y (deg)')
    ax.set_zlabel('f(r)')
    ax = fig.add_subplot(222)
    ax.plot(x[:,y0], pdfs[:,y0], color='r')
    ax.set_xlim([0,xymax])
    # ax.set_xlabel('Distance (deg)')
    ax.set_ylabel('f(r)')
    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(x, y, cdfs, s=0.1, alpha=0.5)
    ax.plot(x[:,y0], y[:,y0], cdfs[:,y0], color='r')
    ax.set_zlim([0,1])
    ax.set_xlabel('x (deg)')
    ax.set_ylabel('y (deg)')
    ax.set_zlabel('F(r)')
    ax = fig.add_subplot(224)
    ax.plot(x[:,y0], cdfs[:,y0], color='r')
    ax.set_xlim([0,xymax])
    ax.set_ylim([0,1])
    ax.set_xlabel('Distance (deg)')
    ax.set_ylabel('F(r)')
    plt.show()
    

def plot_time_pdf(c, p, ta):
    print('app_support.plot_time_pdf')
    sys.stdout.flush()
    fig, ax = plt.subplots(2, 1, sharex=True)
    if ta is None:
        ###################### untruncated time distribution #################
        tmax = min(20, inv_cdf_time(0.995, c, p))
        t = np.arange(0, tmax+0.1, 0.1) # bins
        pdft = pdf_time(t, c, p)
        cdft = cdf_time(t, c, p)
        label='untruncated time pdf\nc='+str(c)+'\np='+str(p)
    else:
        ####################### truncated time distribution ##################
        t = np.arange(0, ta+0.1, 0.1) # bins
        pdft = pdf_time_trunc(t, c, p, ta)
        cdft = cdf_time_trunc(t, c, p, ta)
        label='truncated time pdf\nc='+str(c)+'\np='+str(p)+'\nta='+str(ta)+"yr"
    # pdf
    ax[0].plot(t, pdft, 'r-', lw=2, alpha=0.6, label=label)
    ax[0].set_ylabel('f(t)')
    ax[0].set_yscale("log")
    ax[0].legend()
    # cdf    
    ax[1].plot(t, cdft, 'r-', lw=2, alpha=0.6)
    ax[1].set_ylim([0,1])
    ax[1].set_xlabel('t (days)')
    ax[1].set_ylabel('F(t)')
    plt.show()



def get_default_depth_distribution():
    '''
    default depth distribution
    '''
    dep_distr = list()
    depths = [12., 30.]
    for depth in depths:
        dep_distr.append( (1./len(depths), depth) )
    pmf = PMF(dep_distr)
    return pmf



def get_default_np_distribution():
    '''
    default nodal plane distribution
    '''
    nps = list()
    for strike in [0.]:
        for dip in np.arange(30., 91., 15.):
            for rake in np.arange(-90., 181., 90.):
                nps.append(get_np(strike, dip, rake))
    prob = 1./len(nps)
    np_distr = list()
    for nodal_plane in nps:
        np_distr.append( (prob, nodal_plane) )
    pmf = PMF(np_distr)
    return pmf


def get_np(strike, dip, rake):
    return NodalPlane(strike=strike, dip=dip, rake=rake)


def get_pmf(data):
    return PMF(data)



def plot_time_etas_bulk(simulations, mag_threshold, show_random=True,
                        ax=None, show=False):
    res = get_cumnum_matrix_vs_time(simulations, mag_threshold)
    
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    for te in random.sample(range(len(simulations)), int(0.05*len(simulations))):
        ax.plot(res["time_vec"], res["num_matrix"][te,:], color=[0.5,0.5,0.5],
                linewidth=1., label="50 random simulations")
    ax.fill_between(res["time_vec"], 
                    np.percentile(res["num_matrix"], 2.5, axis=0),
                    np.percentile(res["num_matrix"], 97.5, axis=0), color="r",
                    alpha=.2, label=r"$2.5^{th}-97.5^{th}$ percentile range")
    if show_random:
        ax.plot(res["time_vec"], np.percentile(res["num_matrix"], 50, axis=0), 
                color="r", linewidth=2., label=r'Median simulations')
    handles, labes = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labes, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Cumulative number of events M>='+str(mag_threshold))
    if show:
        plt.show()
    return ax


def get_cumnum_vs_time(simulation, mag_threshold, max_time=None):
    # cumulative number of aftershocks with time
    mags = [rup.mag for rup in simulation.catalog["rupture"]]
    filt = np.array(mags) >= mag_threshold
    bla = simulation.filter(filt)
    if bla.get_num_events() <= 1:
        return {"num_events": [0], "time_vec": np.array([0.])}
    bla.process_catalog_4etas(mag_threshold,
                              sim_start=bla.catalog["datetime"][0])
    if bla.get_num_events() <= 1:
        return {"num_events": [0], "time_vec": np.array([0.])}

    times = bla.catalog["tt"]
    if max_time is None:
        time_vec = np.arange(0., np.ceil(np.max(times)), 0.1)
    else:
        time_vec = np.arange(0., max_time, 0.1)
    histogram = np.histogram(times, time_vec)
    num_events = [0]
    num_events.extend(list(histogram[0].cumsum()))
    return {"num_events": num_events, "time_vec": time_vec}
    

def get_cumnum_matrix_vs_time(simulations, mag_threshold, max_time=None):
    res_list = list()
    for l in range(0, len(simulations)):
        res_list.append(get_cumnum_vs_time(simulations[l], mag_threshold))
    if max_time is None:
        maxt = np.ceil(max([res["time_vec"][-1] for res in res_list]))
        for l, res in enumerate(res_list):
            res["time_vec"][-1]
    else:
        maxt = max_time
    time_vec = np.arange(0., maxt, 0.1)
    num_matrix = np.zeros((len(simulations), time_vec.shape[0]))
    for l in range(0, len(simulations)):
        res = res_list[l]
        num_matrix[l, 0:res["time_vec"].shape[0]] = res["num_events"]
        num_matrix[l, res["time_vec"].shape[0]:] = res["num_events"][-1]
    return {"num_matrix": num_matrix, "time_vec": time_vec}


def plot_time_etas_single(simulation, mag_threshold, ax=None, show=False):
    res = get_cumnum_vs_time(simulation, mag_threshold)
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    ax.plot(res["time_vec"], res["num_events"], color='r', label='One realization')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Cumulative number of events M>='+str(mag_threshold))
    if show:
        plt.show()
    return ax
    
    
def plot_space_etas_single(simulation, mag_threshold, include_basemap_flag=0,
                           show=False):
    mags = [rup.mag for rup in simulation.catalog["rupture"]]
    filt = np.array(mags) >= mag_threshold
    bla = simulation.filter(filt)
    bla.process_catalog_4etas(mag_threshold)

    from matplotlib import gridspec
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1]) 
    # fig, axs = plt.subplots(2, 2, 
    #                         gridspec_kw={'width_ratios': [3,1],
    #                                            'height_ratios' : [3,1]})
    # fig.delaxes(axs[1][1])
    # 
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2], sharex=ax)
    ax3 = fig.add_subplot(gs[1], sharey=ax)

    # ax = axs[0][0]
    extent = [np.floor(min(simulation.catalog['longitude'])/0.5)*0.5,
              np.ceil(max(simulation.catalog['longitude'])/0.5)*0.5,
              np.floor(min(simulation.catalog['latitude'])/0.5)*0.5,
              np.ceil(max(simulation.catalog['latitude'])/0.5)*0.5]
    lon_step = 0.5
    lat_step = 0.5
    meridians = np.round(np.arange(extent[0], extent[1] + lon_step, lon_step), 2)
    parallels = np.round(np.arange(extent[2], extent[3] + lat_step, lat_step), 2)

    if include_basemap_flag == 1:
        from mpl_toolkits.basemap import Basemap
        bm = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[2],
                      urcrnrlon=extent[1], urcrnrlat=extent[3],
                      projection='cyl', resolution='h', fix_aspect=False, ax=ax)
        ax.add_collection(bm.drawcoastlines(linewidth=0.25))
        ax.add_collection(bm.drawcountries(linewidth=0.35))
    
    # data
    df = simulation.get_df()
    df.loc[ df['magnitude'] >= mag_threshold ]
    
    ax.scatter(df['longitude'], df['latitude'], df['magnitude'] ** 2,
               color=[0.5,0.5,0.5], alpha=0.5, label=str(df.shape[0])+" events") 
    df1 = df.loc[df['magnitude'] > 5.95]
    ax.scatter(df1['longitude'], df1['latitude'], df1['magnitude'] ** 2,
               marker="*", c='m', edgecolor='k', linewidth=0.5,
               label='M$_W>$5.95') 
    df2 = df.loc[df['magnitude'] > 6.95]
    ax.scatter(df2['longitude'], df2['latitude'], df2['magnitude'] ** 2,
               marker="*", c='r', edgecolor='k', linewidth=0.5,
               label='M$_W>$6.95') 
    ax.set_ylabel('Latitude')
    ax.set_xticks(meridians)
    ax.set_yticks(parallels)
    # ax.set_xticklabels("")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    # ax2 = axs[1][0]
    ax2.scatter(df['longitude'], df['depth'], df['magnitude'] ** 2,
               color=[0.5,0.5,0.5], alpha=0.5, label=str(df.shape[0])+" events") 
    df1 = df.loc[df['magnitude'] > 5.95]
    ax2.scatter(df1['longitude'], df1['depth'], df1['magnitude'] ** 2,
               marker="*", c='m', edgecolor='k', linewidth=0.5,
               label='M$_W>$5.95') 
    df2 = df.loc[df['magnitude'] > 6.95]
    ax2.scatter(df2['longitude'], df2['depth'], df2['magnitude'] ** 2,
               marker="*", c='r', edgecolor='k', linewidth=0.5,
               label='M$_W>$6.95') 
    ax2.set_ylim(bottom=0.)
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Depth')
    ax2.set_xticks(meridians)
    ax2.set_ylim([ax2.get_ylim()[1], ax2.get_ylim()[0]])
    
    # ax3 = axs[0][1]
    ax3.scatter(df['depth'], df['latitude'], df['magnitude'] ** 2,
               color=[0.5,0.5,0.5], alpha=0.5, label=str(df.shape[0])+" events") 
    df1 = df.loc[df['magnitude'] > 5.95]
    ax3.scatter(df1['depth'], df1['latitude'], df1['magnitude'] ** 2,
               marker="*", c='m', edgecolor='k', linewidth=0.5,
               label='M$_W>$5.95') 
    df2 = df.loc[df['magnitude'] > 6.95]
    ax3.scatter(df2['depth'], df2['latitude'], df2['magnitude'] ** 2,
               marker="*", c='r', edgecolor='k', linewidth=0.5,
               label='M$_W>$6.95')     
    ax3.set_xlim(left=0.)
    ax3.set_ylim(extent[2], extent[3])
    ax3.set_xlabel('Depth')
    ax3.set_yticks(parallels)
    # ax3.set_yticklabels("")

    if show:
        plt.show()
    return fig, (ax, ax2, ax3)



def plot_space_etas_single_1(simulation, mag_threshold, include_basemap_flag=0,
                             show=False):
    mags = [rup.mag for rup in simulation.catalog["rupture"]]
    filt = np.array(mags) >= mag_threshold
    bla = simulation.filter(filt)
    bla.process_catalog_4etas(mag_threshold)

    fig = plt.figure()
    ax = fig.add_subplot()

    extent = [np.floor(min(bla.catalog['longitude'])/0.5)*0.5,
              np.ceil(max(bla.catalog['longitude'])/0.5)*0.5,
              np.floor(min(bla.catalog['latitude'])/0.5)*0.5,
              np.ceil(max(bla.catalog['latitude'])/0.5)*0.5]
    lon_step = 0.5
    lat_step = 0.5
    meridians = np.round(np.arange(extent[0], extent[1] + lon_step, lon_step), 2)
    parallels = np.round(np.arange(extent[2], extent[3] + lat_step, lat_step), 2)

    if include_basemap_flag == 1:
        from mpl_toolkits.basemap import Basemap
        bm = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[2],
                      urcrnrlon=extent[1], urcrnrlat=extent[3],
                      projection='cyl', resolution='h', fix_aspect=False, ax=ax)
        ax.add_collection(bm.drawcoastlines(linewidth=0.25))
        ax.add_collection(bm.drawcountries(linewidth=0.35))
    
    # data
    df = bla.get_df()
    df.loc[ df['magnitude'] >= mag_threshold ]
    
    ax.scatter(df['longitude'], df['latitude'], df['magnitude'] ** 2,
               color=[0.5,0.5,0.5], alpha=0.5, label=str(df.shape[0])+" events") 
    df1 = df.loc[df['magnitude'] > 5.95]
    ax.scatter(df1['longitude'], df1['latitude'], df1['magnitude'] ** 2,
               marker="*", c='m', edgecolor='k', linewidth=0.5,
               label='M$_W>$5.95') 
    df2 = df.loc[df['magnitude'] > 6.95]
    ax.scatter(df2['longitude'], df2['latitude'], df2['magnitude'] ** 2,
               marker="*", c='r', edgecolor='k', linewidth=0.5,
               label='M$_W>$6.95') 
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xticks(meridians)
    ax.set_yticks(parallels)
    # ax.set_xticklabels("")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    if show:
        plt.show()
    return ax




def plot_length_etas_bulk(simulations, mag_threshold, ax=None, show=False):
    maxt = list()
    for l in range(0, len(simulations)):
        dff = simulations[l][ (simulations[l]['magnitude']>=mag_threshold)]
        times = dff['tt']-simulations[l]['tt'].iloc[0]
        maxt.append( np.max(times)/365 )
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    ax.hist(maxt, bins=20)
    ax.set_xlabel('Length of simulations (yr)')
    ax.set_ylabel('Number of simulations')
    if show:
        plt.show()





def plot_space_etas_bulk(simulations, mag_threshold, include_basemap_flag=0,
                         step=0.5, show=False):
    max_lons = list()
    min_lons = list()
    max_lats = list()
    min_lats = list()
    for l in range(0, len(simulations)):
        mags = [rup.mag for rup in simulations[l].catalog["rupture"]]
        filt = np.array(mags) > mag_threshold
        bla = simulations[l].filter(filt)
        bla.process_catalog_4etas(mag_threshold,
                                  sim_start=bla.catalog["datetime"][0])
        max_lons.append(bla.catalog["longitude"].max())
        min_lons.append(bla.catalog["longitude"].min())
        max_lats.append(bla.catalog["latitude"].max())
        min_lats.append(bla.catalog["latitude"].min())
        
    extent = [np.floor(np.min(min_lons)/step)*step,
              np.ceil(np.max(max_lons)/step)*step,
              np.floor(np.min(min_lats)/step)*step,
              np.ceil(np.max(max_lats)/step)*step]
    lon_step = step
    lat_step = step
    meridians = np.round(np.arange(extent[0], extent[1] + lon_step, lon_step), 2)
    parallels = np.round(np.arange(extent[2], extent[3] + lat_step, lat_step), 2)
        
    xbins = np.arange(extent[0], extent[1]+0.1, 0.1)
    ybins = np.arange(extent[2], extent[3]+0.1, 0.1)
    bins = np.zeros((ybins.shape[0]-1, xbins.shape[0]-1))
    
    discarted = list()
    for l in range(0, len(simulations)):
        mags = [rup.mag for rup in simulations[l].catalog["rupture"]]
        filt = np.array(mags) >= mag_threshold
        bla = simulations[l].filter(filt)
        bla.process_catalog_4etas(mag_threshold)
        discarted.append(0.)
        for rup in bla.catalog["rupture"]:
            lon = rup.hypocenter.longitude
            lat = rup.hypocenter.latitude
            if lon > xbins.max() or lon < xbins.min() or \
               lat > ybins.max() or lat < ybins.min():
                discarted[l] += 1.
            else:
                indx = np.where(lon > xbins)[0][-1]
                indy = np.where(lat > ybins)[0][-1]
                bins[indy, indx] += 1.    
    
    fig = plt.figure(figsize=(6*1.2, 6))
    ax = fig.add_subplot()
    
    xx, yy = np.meshgrid(xbins[:-1]+0.05, ybins[:-1]+0.05)
    colormesh = ax.pcolormesh(xx, yy, bins/len(simulations),
                              cmap=cm.viridis, alpha=0.9,
                              norm=mpl.colors.LogNorm())

    ax.set_xlabel('$\mathrm{Longitude\ (^\circ)}$') #, labelpad=15)
    ax.set_ylabel('$\mathrm{Latitude\ (^\circ)}$') #, labelpad=15)

    ax.set_yticks(parallels)
    ax.set_yticklabels(["$"+str(par)+"$" for par in parallels],
                            verticalalignment='center', horizontalalignment='right')
    # ax.set_yticklabels(" ")
    ax.set_xticks(meridians)
    ax.set_xticklabels(["$"+str(mer)+"$" for mer in meridians])
    ax.set_xlim([extent[0]+0.1, extent[1]-0.1])
    ax.set_ylim([extent[2]+0.1, extent[3]-0.1])
    
    cb = plt.colorbar(colormesh,
                      label='$Rate\ per\ 0.1^\circ\ x\ 0.1^\circ}$')
    ax.tick_params(axis='both', which='major', pad=7)
 
    if show:
        plt.show()


#     temp = list()
#     time_vec = np.arange(0., max(maxt), 0.1)
#     for l in range(0, len(simulations)):
#         mags = [rup.mag for rup in simulations[l].catalog["rupture"]]
#         filt = np.array(mags) > mag_threshold
#         bla = simulations[l].filter(filt)
#         bla.process_catalog_4etas(mag_threshold,
#                                   sim_start=bla.catalog["datetime"][0])
#         times = bla.catalog["tt"]
#         histogram = np.histogram(times, time_vec)
#         num_events = [0]
#         num_events.extend(list(histogram[0].cumsum()))
#         temp.append( num_events )
#     for te in random.sample(temp, int(0.05*len(simulations))):
#         ax.plot(time_vec, te, color=[0.5,0.5,0.5], linewidth=1.,
#                 label="50 random simulations")
#     ax.fill_between(time_vec, np.percentile(temp, 2.5, axis=0),
#                     np.percentile(temp, 97.5, axis=0), color="r",
#                     alpha=.2, label=r"$2.5^{th}-97.5^{th}$ percentile range")
#     ax.plot(time_vec, np.percentile(temp, 50, axis=0), color="r", linewidth=2.,
#             label=r'Median simulations')
#     handles, labes = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labes, handles))
#     ax.legend(by_label.values(), by_label.keys())
#     ax.set_xlabel('Time (days)')
#     ax.set_ylabel('Cumulative number of events M>='+str(mag_threshold))
#     if show:
#         plt.show()
#     return ax):

   
#     xx, yy = np.meshgrid(data["region"]["lon"]+0.05,
#                          data["region"]["lat"]+0.05)
#     colormesh = bm.pcolormesh(xx, yy, data["count"], cmap=cm.viridis,
#                               norm=mpl.colors.LogNorm())

#     ax.scatter(data["catalog"]["lon"], data["catalog"]["lat"],
#                color="k", alpha=0.1) #marker=marker, color=col, s=size)
    
#     ax.set_xlim([extent[0]+0.1, extent[1]-0.1])
#     ax.set_ylim([extent[2]+0.1, extent[3]-0.1])
    


#     # m = cm.ScalarMappable(cmap=getattr(cm, "viridis"))
#     # m.set_array(np.log(data["count"]))
#     # # m.set_clim(vmin=-max([-min(data), max(data)]),
#     # #             vmax=max([-min(data), max(data)]))
#     # colors = m.to_rgba(np.log(data["count"]))
#     # # colorbar
#     # # where arg is [left, bottom, width, height]
#     # cb_ax = fig.add_axes([ .81, .24, .04, .545])
#     # # cb_ax = fig.add_axes([0.24, 0.10, 0.545, 0.04])
#     # cb = fig.colorbar(m, cax=cb_ax, orientation="vertical")
#     # cb.set_label(label='$M_W4+\mathrm{\ rate\ per\ 0.1^\circ\ x\ 0.1^\circ\ per\ 3\ years}$') #, weight='bold')

#     ax.tick_params(axis='both', which='major', pad=7)
 
#     plt.show()
#     plt.savefig("fig_kaikoura_contour.png",
#                 bbox_inches='tight', dpi=600, format="png")



#     simulation.process_catalog_4etas(mag_threshold)

#     from matplotlib import gridspec
#     fig = plt.figure()
#     gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1]) 
#     # fig, axs = plt.subplots(2, 2, 
#     #                         gridspec_kw={'width_ratios': [3,1],
#     #                                            'height_ratios' : [3,1]})
#     # fig.delaxes(axs[1][1])
#     # 
#     ax = fig.add_subplot(gs[0])
#     ax2 = fig.add_subplot(gs[2], sharex=ax)
#     ax3 = fig.add_subplot(gs[1], sharey=ax)

#     # ax = axs[0][0]
#     extent = [np.floor(min(simulation.catalog['longitude'])/0.5)*0.5,
#               np.ceil(max(simulation.catalog['longitude'])/0.5)*0.5,
#               np.floor(min(simulation.catalog['latitude'])/0.5)*0.5,
#               np.ceil(max(simulation.catalog['latitude'])/0.5)*0.5]
#     lon_step = 0.5
#     lat_step = 0.5
#     meridians = np.round(np.arange(extent[0], extent[1] + lon_step, lon_step), 2)
#     parallels = np.round(np.arange(extent[2], extent[3] + lat_step, lat_step), 2)

#     if include_basemap_flag == 1:
#         from mpl_toolkits.basemap import Basemap
#         bm = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[2],
#                       urcrnrlon=extent[1], urcrnrlat=extent[3],
#                       projection='cyl', resolution='h', fix_aspect=False, ax=ax)
#         ax.add_collection(bm.drawcoastlines(linewidth=0.25))
#         ax.add_collection(bm.drawcountries(linewidth=0.35))
    
    
#     df = simulation[ (simulation['magnitude'] > mag_threshold) ]
#     ax.scatter(df['longitude'], df['latitude'], df['magnitude'] ** 2,
#                color=[0.5,0.5,0.5], alpha=0.5, label=str(df.shape[0])+" events") 
#     df1 = df.loc[simulation['magnitude'] > 5.95]
#     ax.scatter(df1['longitude'], df1['latitude'], df1['magnitude'] ** 2,
#                marker="*", c='m', edgecolor='k', linewidth=0.5,
#                label='M$_W>$5.95') 
#     df2 = df.loc[simulation['magnitude'] > 6.95]
#     ax.scatter(df2['longitude'], df2['latitude'], df2['magnitude'] ** 2,
#                marker="*", c='r', edgecolor='k', linewidth=0.5,
#                label='M$_W>$6.95') 
#     ax.set_ylabel('Latitude')
#     ax.set_xticks(meridians)
#     ax.set_yticks(parallels)
#     # ax.set_xticklabels("")
#     ax.set_xlim(extent[0], extent[1])
#     ax.set_ylim(extent[2], extent[3])
    
#     # ax2 = axs[1][0]
#     df = simulation[ (simulation['magnitude'] > mag_threshold) ]
#     ax2.scatter(df['longitude'], df['depth'], df['magnitude'] ** 2,
#                color=[0.5,0.5,0.5], alpha=0.5, label=str(df.shape[0])+" events") 
#     df1 = df.loc[simulation['magnitude'] > 5.95]
#     ax2.scatter(df1['longitude'], df1['depth'], df1['magnitude'] ** 2,
#                marker="*", c='m', edgecolor='k', linewidth=0.5,
#                label='M$_W>$5.95') 
#     df2 = df.loc[simulation['magnitude'] > 6.95]
#     ax2.scatter(df2['longitude'], df2['depth'], df2['magnitude'] ** 2,
#                marker="*", c='r', edgecolor='k', linewidth=0.5,
#                label='M$_W>$6.95') 
#     ax2.set_ylim(bottom=0.)
#     ax2.set_xlim(extent[0], extent[1])
#     ax2.set_xlabel('Longitude')
#     ax2.set_ylabel('Depth')
#     ax2.set_xticks(meridians)
#     ax2.set_ylim([ax2.get_ylim()[1], ax2.get_ylim()[0]])
    
#     # ax3 = axs[0][1]
#     df = simulation[ (simulation.catalog['magnitude'] > mag_threshold) ]
#     ax3.scatter(df['depth'], df['latitude'], df['magnitude'] ** 2,
#                color=[0.5,0.5,0.5], alpha=0.5, label=str(df.shape[0])+" events") 
#     df1 = df.loc[simulation.catalog['magnitude'] > 5.95]
#     ax3.scatter(df1['depth'], df1['latitude'], df1['magnitude'] ** 2,
#                marker="*", c='m', edgecolor='k', linewidth=0.5,
#                label='M$_W>$5.95') 
#     df2 = df.loc[simulation.catalog['magnitude'] > 6.95]
#     ax3.scatter(df2['depth'], df2['latitude'], df2['magnitude'] ** 2,
#                marker="*", c='r', edgecolor='k', linewidth=0.5,
#                label='M$_W>$6.95')     
#     ax3.set_xlim(left=0.)
#     ax3.set_ylim(extent[2], extent[3])
#     ax3.set_xlabel('Depth')
#     ax3.set_yticks(parallels)
#     # ax3.set_yticklabels("")

#     if show:
#         plt.show()

#         count = forecast.expected_rates.spatial_counts(cartesian=True)
#         region = forecast.expected_rates.region        
#         from myutils.utils_pickle import save_pickle
#         aaa = {"count": count,
#                "region": {"lon": region.xs, "lat": region.ys},
#                "catalog": {"lon": lon, "lat": lat}}
#         save_pickle(aaa, "pick_plot_events")
        
        
        
#         cb = bm.colorbar(colormesh, location='bottom', pad=1.,
#                       label='$M_W4+\mathrm{\ rate\ per\ 0.1^\circ\ x\ 0.1^\circ\ per\ 3\ years}$')
