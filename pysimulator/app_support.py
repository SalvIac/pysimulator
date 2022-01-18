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


import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True




import os
sys.path.append('C:\\Users\\Salvatore\\Dropbox\\SalvIac')
import datetime
import numpy as np
import pandas as pd
import support_functions
from rupture_builder import RuptureBuilder
from custom_catalog import CustomCatalog
from exposure_builder import ExposureBuilder
from etas_simulator import EtasSimulator
from myutils.utils_pickle import load_pickle, save_pickle
from openquake.hazardlib.gsim.base import registry



def set_Tk_var():
    global add_rupture_geometries_flag
    add_rupture_geometries_flag = tk.IntVar(value=1)
    global include_basemap_flag1
    include_basemap_flag1 = tk.IntVar()
    global entry_mesh_spacing
    entry_mesh_spacing = tk.StringVar(value="1.")
    global combobox_geometry_exp
    combobox_geometry_exp = tk.StringVar()
    global exp_num_assets
    exp_num_assets = tk.StringVar(value="1.")
    global include_basemap_flag2
    include_basemap_flag2 = tk.IntVar()
    global c1
    c1 = tk.StringVar(value="1.")
    global c2
    c2 = tk.StringVar(value="4.5")
    global c3
    c3 = tk.StringVar(value="0.75")
    global m_min_inc
    m_min_inc = tk.StringVar(value="6.")
    global combobox_timetrunc
    combobox_timetrunc = tk.StringVar()
    global combobox_magtrunc
    combobox_magtrunc = tk.StringVar()
    global combobox_spatialpdf
    combobox_spatialpdf = tk.StringVar()
    global combobox_maginc
    combobox_maginc = tk.StringVar()
    global combobox_multiproces
    combobox_multiproces = tk.StringVar()
    global combobox_etas_analyses
    combobox_etas_analyses = tk.StringVar()
    global include_basemap3
    include_basemap3 = tk.IntVar()
    global include_exposure_etas
    include_exposure_etas = tk.IntVar()
    global incldue_geom_etas
    incldue_geom_etas = tk.IntVar()
    global include_exposure_etas2
    include_exposure_etas2 = tk.IntVar()
    global combobox_available_gmm
    combobox_available_gmm = tk.StringVar()
    global combobox_simul_id_gmm
    combobox_simul_id_gmm = tk.StringVar()
    global combobox_event_id_gmm
    combobox_event_id_gmm = tk.StringVar()
    global combobox_imt_gmm
    combobox_imt_gmm = tk.StringVar()
    global combobox
    combobox = tk.StringVar()
    
def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top
    # combobox_available_gmm.config(values=[a[0] for a in sorted(registry.items())])




##############################################################################
def add_depth_pdf():
    print('app_support.add_depth_pdf')
    sys.stdout.flush()
    depth = support_functions.string2float(w.EntryDepthEtas.get())
    prob = support_functions.string2float(w.EntryDepthProb.get())
    print(prob, depth)
    index = len(w.ScrolledtreeviewDepthEtas.get_children())
    w.ScrolledtreeviewDepthEtas.insert(parent="", index="end", iid=index,
                                    values=(prob, depth))
    # clear boxes
    w.EntryDepthEtas.delete(0,"end")
    w.EntryDepthProb.delete(0,"end")
##############################################################################





def add_gmm():
    print('app_support.add_gmm')
    sys.stdout.flush()






##############################################################################
def add_nodal_distr():
    print('app_support.add_nodal_distr')
    sys.stdout.flush()
    strike = support_functions.string2float(w.EntryStrikeEtas.get())
    dip = support_functions.string2float(w.EntryDipEtas.get())
    rake = support_functions.string2float(w.EntryRakeEtas.get())
    prob = support_functions.string2float(w.EntryProbEtas.get())
    print(prob, strike, dip, rake)
    index = len(w.ScrolledtreeviewNPEtas.get_children())
    w.ScrolledtreeviewNPEtas.insert(parent="", index="end", iid=index,
                                    values=(prob, strike, dip, rake))
    # clear boxes
    w.EntryStrikeEtas.delete(0,"end")
    w.EntryDipEtas.delete(0,"end")
    w.EntryRakeEtas.delete(0,"end")
    w.EntryProbEtas.delete(0,"end")


def add_taxonomy():
    print('app_support.add_taxonomy')
    sys.stdout.flush()
    tax_id = w.EntryTaxId.get()
    if tax_id == "":
        raise Exception("taxonomy id empty!")
    tax_perc = support_functions.string2float(w.EntryTaxPerc.get())
    print(tax_id, tax_perc)
    index = len(w.ScrolledtreeviewTax.get_children())
    w.ScrolledtreeviewTax.insert(parent="", index="end", iid=index,
                                 values=(tax_id, tax_perc))
    # clear boxes
    w.EntryTaxId.delete(0,"end")
    w.EntryTaxPerc.delete(0,"end")
    

def build_catalog():
    print('app_support.build_catalog')
    sys.stdout.flush()
    ruptures = list()
    datetimes = list()
    mesh_spacing = support_functions.string2float(w.EntryMeshRuptures.get())
    for child in w.ScrolledtreeviewCatalog.get_children():
        item = w.ScrolledtreeviewCatalog.item(child, "values")
        print(item)
        datetimes.append(datetime.datetime.strptime(item[4], "%Y-%m-%d %H:%M:%S"))
        ruptures.append( RuptureBuilder.init_surface_from_point(
                                        support_functions.string2float(item[0]),
                                        support_functions.string2float(item[1]),
                                        support_functions.string2float(item[2]),
                                        support_functions.string2float(item[3]),
                                        support_functions.string2float(item[5]),
                                        support_functions.string2float(item[6]),
                                        support_functions.string2float(item[7]),
                                        mesh_spacing=mesh_spacing) )
    global catalog_to_start
    catalog_to_start = CustomCatalog.from_ruptures(datetimes, ruptures)
    print(catalog_to_start)
    # activate buttons
    w.ButtonPlot2D.config(state="normal")
    w.ButtonPlot3D.config(state="normal")
    # w.ButtonWebmap.config(state="normal")
    w.CheckbuttonBasemap1.config(state="normal")
    w.ButtonBuildEtas.config(state="normal")
##############################################################################














def build_catalog_from_oq():
    print('app_support.build_catalog_from_oq')
    sys.stdout.flush()
    
    
    
    
    
    
    
    
    
    
    
    
    
##############################################################################
def build_etas_model():
    print('app_support.build_etas_model')
    sys.stdout.flush()

    b = support_functions.string2float(w.EntryB.get())
    mmin = support_functions.string2float(w.EntryMmin.get())
    mmax = support_functions.string2float(w.EntryMmax.get())
    c1 = support_functions.string2float(w.EntryC1.get())
    c2 = support_functions.string2float(w.EntryC2.get())
    c3 = support_functions.string2float(w.EntryC3.get())
    m_min_inc = support_functions.string2float(w.EntryMminInc.get())
    A = support_functions.string2float(w.EntryA.get())
    alpha = support_functions.string2float(w.EntryAlpha.get())
    q = support_functions.string2float(w.EntryQ.get())
    D = support_functions.string2float(w.EntryD.get())
    gamma = support_functions.string2float(w.EntryGamma.get())
    c = support_functions.string2float(w.EntryC.get())
    p = support_functions.string2float(w.EntryP.get())
    ta = support_functions.string2float(w.EntryTa.get())
    timetrunc = combobox_timetrunc.get()
    magtrunc = combobox_magtrunc.get()
    spatialpdf = "5" #combobox_spatialpdf.get()
    incompletess = combobox_maginc.get()
    num_simul = int(w.EntryNumSimul.get())
    multi = True if combobox_multiproces.get() == "True" else False
    
    params = {
        "alpha": alpha, "A": A,
        "p": p, "c": c,
        "D": D, "q": q, "gamma": gamma,
        "b": b, "min_mag": mmin, "max_mag": mmax,
               }
    if (ta is None) and (combobox_timetrunc.get() == "True"):
        raise Exception("time truncation true but no ta value provided!")
    else:
        params["ta"] = ta*365
    if combobox_maginc.get() == "True":
        params["c1"] = c1
        params["c2"] = c2
        params["c3"] = c3
        params["incompl_min_mag"] = m_min_inc
              
    model = "timetrunc:"+timetrunc+"_magnitudetrunc:"+magtrunc+\
            "_spatialpdf:"+spatialpdf+"_incompletess:"+incompletess
    options = {"multiprocessing": multi, "num_realization":num_simul}

    # nodal plane distribution
    if len(w.ScrolledtreeviewNPEtas.get_children()) == 0:
        np_distr = support_functions.get_default_np_distribution()
    else:
        _sum = sum([support_functions.string2float(w.ScrolledtreeviewNPEtas.item(child, "values")[0])
                    for child in w.ScrolledtreeviewNPEtas.get_children()])
        if not np.equal(1., _sum):
            print("nodal planes re-normalized")
        np_distr = list()  
        for child in w.ScrolledtreeviewNPEtas.get_children():
            item = w.ScrolledtreeviewNPEtas.item(child, "values")
            np_distr.append( [support_functions.string2float(item[0])/_sum,
                              support_functions.get_np(
                                  support_functions.string2float(item[1]),
                                  support_functions.string2float(item[2]),
                                  support_functions.string2float(item[3])) ] )
        np_distr = support_functions.get_pmf(np_distr)
        
    # depth distribution
    if len(w.ScrolledtreeviewDepthEtas.get_children()) == 0:
        dep_distr = support_functions.get_default_depth_distribution()
    else:
        _sum = sum([support_functions.string2float(w.ScrolledtreeviewDepthEtas.item(child, "values")[0])
                    for child in w.ScrolledtreeviewDepthEtas.get_children()])
        if not np.equal(1., _sum):
            print("depths re-normalized")
        dep_distr = list()  
        for child in w.ScrolledtreeviewDepthEtas.get_children():
            item = w.ScrolledtreeviewDepthEtas.item(child, "values")
            dep_distr.append( [support_functions.string2float(item[0])/_sum,
                               support_functions.string2float(item[1]) ] )
        dep_distr = support_functions.get_pmf(dep_distr)
    
    print(params)
    print(model)
    print(options)
    print(np_distr)
    global etas_sim
    etas_sim = EtasSimulator(params, catalog_to_start, model,
                             simul_options=options,
                             nodal_planes_distr=np_distr,
                             depth_distr=dep_distr)
    w.ButtonRunETAS.config(state="normal")


def build_exposure():
    print('app_support.build_exposure')
    sys.stdout.flush()
    clon = support_functions.string2float(w.EntryCLon.get())
    clat = support_functions.string2float(w.EntryCLat.get())
    lon_width = support_functions.string2float(w.EntryWLon.get())
    lat_height = support_functions.string2float(w.EntryHLat.get())
    lon_bin = support_functions.string2float(w.EntryBinLon.get())
    lat_bin = support_functions.string2float(w.EntryBinLat.get())
    geometry = combobox_geometry_exp.get()
    number = support_functions.string2float(w.EntryNum.get())
    taxonomy = dict()
    for child in w.ScrolledtreeviewTax.get_children():
        item = w.ScrolledtreeviewTax.item(child, "values")
        print(item)
        taxonomy[item[0]] = support_functions.string2float(item[1])
    # normalize taxonomy percentages
    if not np.equal(1., sum([taxonomy[tax] for tax in taxonomy.keys()])):
        _sum = sum([taxonomy[tax] for tax in taxonomy.keys()])
        for tax in taxonomy.keys():
            taxonomy[tax] = taxonomy[tax]/_sum
        print("taxonomy re-normalized")
    print(clon, clat, lon_width, lat_height, lon_bin, lat_bin,
          taxonomy, number, geometry)
    print(geometry, geometry=="Circle")
    global exposure
    if geometry=="Single Point":
        exposure = ExposureBuilder.from_point(clon, clat,
                                              taxonomy, number, geometry)
    else:
        exposure = ExposureBuilder.from_inputs(clon, clat,
                                               lon_width, lat_height, 
                                               lon_bin, lat_bin,
                                               taxonomy, number, geometry)
    # activate other buttons
    w.ButtonPlot2dExpOnly.config(state="normal")
    w.ButtonPlot3dExpOnly.config(state="normal")
    w.ButtonPlot2dWithCat.config(state="normal")
    w.ButtonPlot3dWithCat.config(state="normal")
    w.CheckbuttonBasemap2.config(state="normal")
##############################################################################










def build_exposure_from_oq():
    print('app_support.build_exposure_from_oq')
    sys.stdout.flush()


def build_gmm():
    print('app_support.build_gmm')
    sys.stdout.flush()
    
    






##############################################################################
def catalog_add_event():
    print('app_support.catalog_add_event')
    sys.stdout.flush()
    mag = support_functions.string2float(w.EntryMag.get())#float(w.EntryMag.get())
    lon = support_functions.string2float(w.EntryLong.get())#float(w.EntryLong.get())
    lat = support_functions.string2float(w.EntryLat.get())#float(w.EntryLat.get())
    dep = support_functions.string2float(w.EntryDep.get())#float(w.EntryDep.get())
    dttm = datetime.datetime.strptime(w.EntryDT.get(), "%Y-%m-%d %H:%M:%S")
    stk = support_functions.string2float(w.EntryStrik.get())#float(w.EntryStrik.get())
    dip = support_functions.string2float(w.EntryDi.get())#float(w.EntryDi.get())
    rake = support_functions.string2float(w.EntryRak.get())#float(w.EntryRak.get())
    print(mag, lon, lat, dep, dttm, stk, dip, rake)
    index = len(w.ScrolledtreeviewCatalog.get_children())
    w.ScrolledtreeviewCatalog.insert(parent="", index="end", iid=index,
                                     values=(mag, lon, lat, dep, dttm, stk, dip, rake))
    # clear boxes
    w.EntryMag.delete(0,"end")
    w.EntryLong.delete(0,"end")
    w.EntryLat.delete(0,"end")
    w.EntryDep.delete(0,"end")
    w.EntryDT.delete(0,"end")
    w.EntryStrik.delete(0,"end")
    w.EntryDi.delete(0,"end")
    w.EntryRak.delete(0,"end")
    # activate buttons
    w.ButtonBuildCatalog.config(state="normal")
    w.CheckbuttonAddGeometries.config(state="normal")
    w.LabelMeshRuptures.config(state="normal")
    w.EntryMeshRuptures.config(state="normal")


def catalog_delete_all_events():
    print('app_support.catalog_delete_all_events')
    sys.stdout.flush()
    for child in w.ScrolledtreeviewCatalog.get_children():
        w.ScrolledtreeviewCatalog.delete(child)
    w.ButtonBuildCatalog.config(state="disabled")
    w.CheckbuttonAddGeometries.config(state="disabled")
    w.LabelMeshRuptures.config(state="disabled")
    w.EntryMeshRuptures.config(state="disabled")
    

def catalog_delete_selected_events():
    print('app_support.catalog_delete_selected_events')
    sys.stdout.flush()
    for child in reversed(w.ScrolledtreeviewCatalog.selection()):
        w.ScrolledtreeviewCatalog.delete(child)
    if len(w.ScrolledtreeviewCatalog.get_children()) == 0:
        w.ButtonBuildCatalog.config(state="disabled")
        w.CheckbuttonAddGeometries.config(state="disabled")
        w.LabelMeshRuptures.config(state="disabled")
        w.EntryMeshRuptures.config(state="disabled")


def delete_all_gmm():
    print('app_support.delete_all_gmm')
    sys.stdout.flush()


def delete_all_taxonomy():
    print('app_support.delete_all_taxonomy')
    sys.stdout.flush()
    for child in w.ScrolledtreeviewTax.get_children():
        w.ScrolledtreeviewTax.delete(child)
    

def delete_selected_depth():
    print('app_support.delete_selected_depth')
    sys.stdout.flush()
    for child in reversed(w.ScrolledtreeviewDepthEtas.selection()):
        w.ScrolledtreeviewDepthEtas.delete(child)
##############################################################################







def delete_selected_gmm():
    print('app_support.delete_selected_gmm')
    sys.stdout.flush()







##############################################################################
def delete_selected_nodal():
    print('app_support.delete_selected_nodal')
    sys.stdout.flush()
    for child in reversed(w.ScrolledtreeviewNPEtas.selection()):
        w.ScrolledtreeviewNPEtas.delete(child)
    
     
def delete_selected_taxonomy():
    print('app_support.delete_selected_taxonomy')
    sys.stdout.flush()
    for child in reversed(w.ScrolledtreeviewTax.selection()):
        w.ScrolledtreeviewTax.delete(child)
##############################################################################












def export_csv_etas_bulk():
    print('app_support.export_csv_etas_bulk')
    sys.stdout.flush()


def export_csv_etas_single():
    print('app_support.export_csv_etas_single')
    sys.stdout.flush()














##############################################################################
def plot_2d_catalog():
    print('app_support.plot_2d_catalog')
    sys.stdout.flush()
    support_functions.plot_2d_generic(catalog_to_start, None,
                                      include_basemap_flag1.get(),
                                      show=True)


def plot_2d_exposure():
    print('app_support.plot_2d_exposure')
    sys.stdout.flush()
    support_functions.plot_2d_generic(None, exposure,
                                      include_basemap_flag2.get(),
                                      show=True)


def plot_2d_exposure_catalog():
    print('app_support.plot_2d_exposure_catalog')
    sys.stdout.flush()
    support_functions.plot_2d_generic(catalog_to_start, exposure,
                                      include_basemap_flag2.get(),
                                      show=True)


def plot_3d_catalog():
    print('app_support.plot_3d_catalog')
    sys.stdout.flush()
    support_functions.plot_3d_generic(catalog_to_start, None,
                                      include_basemap_flag1.get(),
                                      show=True)
#############################################################################













def plot_3d_etas():
    print('app_support.plot_3d_etas')
    sys.stdout.flush()
    
    
    
    
    
    
    
    
    
    
#############################################################################
def plot_3d_exposure():
    print('app_support.plot_3d_exposure')
    sys.stdout.flush()
    support_functions.plot_3d_generic(None, exposure,
                                      include_basemap_flag2.get(),
                                      show=True)


def plot_3d_exposure_catalog():
    print('app_support.plot_3d_exposure_catalog')
    sys.stdout.flush()
    support_functions.plot_3d_generic(catalog_to_start, exposure,
                                      include_basemap_flag2.get(),
                                      show=True)
##############################################################################













def plot_animation_etas_bulk():
    print('app_support.plot_animation_etas_bulk')
    sys.stdout.flush()

def plot_animation_etas_single():
    print('app_support.plot_animation_etas_single')
    sys.stdout.flush()

def plot_gmm():
    print('app_support.plot_gmm')
    sys.stdout.flush()
    
    









##############################################################################
def plot_length_etas_bulk():
    print('app_support.plot_length_etas_bulk')
    sys.stdout.flush()
    mag_threshold = support_functions.string2float(w.EntryMinMagPlotsEtas2.get())
    support_functions.plot_length_etas_bulk(etas_sim.output, mag_threshold,
                                            show=True)


def plot_mag_freq_distr():
    print('app_support.plot_mag_freq_distr')
    sys.stdout.flush()
    b = support_functions.string2float(w.EntryB.get())
    mmin = support_functions.string2float(w.EntryMmin.get())
    mmax = support_functions.string2float(w.EntryMmax.get())
    support_functions.plot_mag_freq_distr(b, mmin, mmax)


def plot_mag_incompl():
    print('app_support.plot_mag_incompl')
    sys.stdout.flush()
    c1 = support_functions.string2float(w.EntryC1.get())
    c2 = support_functions.string2float(w.EntryC2.get())
    c3 = support_functions.string2float(w.EntryC3.get())
    m_min_inc = support_functions.string2float(w.EntryMminInc.get())
    support_functions.plot_mag_incompl(m_min_inc, [c1, c2, c3])
    

def plot_productivity():
    print('app_support.plot_productivity')
    sys.stdout.flush()
    A = support_functions.string2float(w.EntryA.get())
    alpha = support_functions.string2float(w.EntryAlpha.get())
    mmin = support_functions.string2float(w.EntryMmin.get())
    support_functions.plot_productivity(A, alpha, mmin)
##############################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
def plot_space_etas_bulk():
    print('app_support.plot_space_etas_bulk')
    sys.stdout.flush()
    mag_threshold = support_functions.string2float(w.EntryMinMagPlotsEtas2.get())


    











##############################################################################
def plot_space_etas_single():
    print('app_support.plot_space_etas_single')
    sys.stdout.flush()    
    index = int(combobox_etas_analyses.get())
    simulations = etas_sim.output[index]
    mag_threshold = support_functions.string2float(w.EntryMinMagPlotsEtas1.get())
    support_functions.plot_space_etas_single(simulations, mag_threshold,
                                             include_basemap3.get(), show=True)


def plot_space_pdf():
    print('app_support.plot_space_pdf')
    sys.stdout.flush()
    q = support_functions.string2float(w.EntryQ.get())
    D = support_functions.string2float(w.EntryD.get())
    gamma = support_functions.string2float(w.EntryGamma.get())
    mmin = support_functions.string2float(w.EntryMmin.get())
    support_functions.plot_space_pdf(q, D, gamma, mmin)
    
    
def plot_time_etas_bulk():
    print('app_support.plot_time_etas_bulk')
    sys.stdout.flush()
    mag_threshold = support_functions.string2float(w.EntryMinMagPlotsEtas2.get())
    support_functions.plot_time_etas_bulk(etas_sim.output, mag_threshold,
                                          show=True)

    
def plot_time_etas_single():
    print('app_support.plot_time_etas_single')
    sys.stdout.flush()
    index = int(combobox_etas_analyses.get())
    simulations = etas_sim.output[index]
    mag_threshold = support_functions.string2float(w.EntryMinMagPlotsEtas1.get())
    support_functions.plot_time_etas_single(simulations, mag_threshold,
                                            show=True)


def plot_time_pdf():
    print('app_support.plot_time_pdf')
    sys.stdout.flush()
    c = support_functions.string2float(w.EntryC.get())
    p = support_functions.string2float(w.EntryP.get())
    ta = support_functions.string2float(w.EntryTa.get())
    support_functions.plot_time_pdf(c, p, ta*365)
##############################################################################







def plot_webmap_catalog():
    print('app_support.plot_webmap_catalog')
    sys.stdout.flush()







def run_etas():
    print('app_support.run_etas')
    sys.stdout.flush()
    out = etas_sim.simulate()
    print(out)
    save_pickle(etas_sim, "etas_sim_test")
    # activate buttons
    w.TComboboxEtasAnalyses.config(state="normal")
    w.TComboboxEtasAnalyses.config(values=[str(i) for i in range(0,len(out))])
    w.ButtonTimePlot1.config(state="normal")
    w.ButtonSpacePlot1.config(state="normal")
    # w.ButtonAnimation.config(state="normal")
    # w.ButtonExportCsv.config(state="normal")
    w.CheckbuttonBasemap3.config(state="normal")
    # w.CheckbuttonExpEtas.config(state="normal")
    # w.CheckbuttonGeomEtas.config(state="normal")
    w.LabelMinMagPlotsEtas1.config(state="normal")
    w.EntryMinMagPlotsEtas1.config(state="normal")
    w.ButtonTimePlot2.config(state="normal")
    w.ButtonSpacePlot2.config(state="normal")
    # w.ButtonPlot3DEtas(state="normal")
    # w.ButtonExportAllCsv.config(state="normal")
    w.ButtonLengthPlot.config(state="normal")
    # w.CheckbuttonGeomEtas2.config(state="normal")
    # w.ButtonAnimation2.config(state="normal")
    w.LabelMinMagPlotsEtas2.config(state="normal")
    w.EntryMinMagPlotsEtas2.config(state="normal")









##############################################################################
def upload_csv_catalog():
    print('app_support.upload_csv_catalog')
    sys.stdout.flush()
    from tkinter import filedialog
    filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                          title="Select a csv file",
                                          filetypes=(("*.csv files", "*.csv"),))
    print(filename)
    df = pd.read_csv(filename)
    df = df.where(df.notnull(), None)
    print(df)
    for index, row in df.iterrows():
        w.ScrolledtreeviewCatalog.insert(parent="", index="end", iid=index,
                                         values=(row["mag"], row["lon"],
                                                 row["lat"], row["dep"],
                                                 datetime.datetime.strptime(
                                                     row["datetime"],
                                                     "%Y-%m-%d %H:%M:%S"),
                                                 row["strike"],
                                                 row["dip"], row["rake"]))
    w.ButtonBuildCatalog.config(state="normal")
    w.CheckbuttonAddGeometries.config(state="normal")
    w.LabelMeshRuptures.config(state="normal")
    w.EntryMeshRuptures.config(state="normal")


def upload_csv_depth():
    print('app_support.upload_csv_depth')
    sys.stdout.flush()
    from tkinter import filedialog
    filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                          title="Select a csv file",
                                          filetypes=(("*.csv files", "*.csv"),))
    print(filename)
    df = pd.read_csv(filename)
    print(df)
    for index, row in df.iterrows():
        w.ScrolledtreeviewDepthEtas.insert(parent="", index="end", iid=index,
                                           values=(row["prob"], row["depth"]))

##############################################################################






def upload_csv_gmm():
    print('app_support.upload_csv_gmm')
    sys.stdout.flush()










def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import app
    app.vp_start_gui()




