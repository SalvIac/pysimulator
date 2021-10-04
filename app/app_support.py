#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 6.2
#  in conjunction with Tcl version 8.6
#    Jun 20, 2021 10:33:44 PM BST  platform: Windows NT

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

def set_Tk_var():
    global add_rupture_geometries_flag
    add_rupture_geometries_flag = tk.IntVar()
    global include_basemap_flag1
    include_basemap_flag1 = tk.IntVar()
    global entry_mesh_spacing
    entry_mesh_spacing = tk.StringVar()
    global combobox_geometry_exp
    combobox_geometry_exp = tk.StringVar()
    global exp_num_assets
    exp_num_assets = tk.StringVar()
    global include_basemap_flag2
    include_basemap_flag2 = tk.IntVar()
    global c1
    c1 = tk.StringVar()
    global c2
    c2 = tk.StringVar()
    global c3
    c3 = tk.StringVar()
    global m_min_inc
    m_min_inc = tk.StringVar()
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

def add_depth_pdf():
    print('app_support.add_depth_pdf')
    sys.stdout.flush()

def add_gmm():
    print('app_support.add_gmm')
    sys.stdout.flush()

def add_nodal_distr():
    print('app_support.add_nodal_distr')
    sys.stdout.flush()

def add_taxonomy():
    print('app_support.add_taxonomy')
    sys.stdout.flush()

def build_catalog():
    print('app_support.build_catalog')
    sys.stdout.flush()

def build_catalog_from_oq():
    print('app_support.build_catalog_from_oq')
    sys.stdout.flush()

def build_etas_model():
    print('app_support.build_etas_model')
    sys.stdout.flush()

def build_exposure():
    print('app_support.build_exposure')
    sys.stdout.flush()

def build_exposure_from_oq():
    print('app_support.build_exposure_from_oq')
    sys.stdout.flush()

def build_gmm():
    print('app_support.build_gmm')
    sys.stdout.flush()

def catalog_add_event():
    print('app_support.catalog_add_event')
    sys.stdout.flush()

def catalog_delete_all_events():
    print('app_support.catalog_delete_all_events')
    sys.stdout.flush()

def catalog_delete_selected_events():
    print('app_support.catalog_delete_selected_events')
    sys.stdout.flush()

def delete_all_gmm():
    print('app_support.delete_all_gmm')
    sys.stdout.flush()

def delete_all_taxonomy():
    print('app_support.delete_all_taxonomy')
    sys.stdout.flush()

def delete_selected_depth():
    print('app_support.delete_selected_depth')
    sys.stdout.flush()

def delete_selected_gmm():
    print('app_support.delete_selected_gmm')
    sys.stdout.flush()

def delete_selected_nodal():
    print('app_support.delete_selected_nodal')
    sys.stdout.flush()

def delete_selected_taxonomy():
    print('app_support.delete_selected_taxonomy')
    sys.stdout.flush()

def export_csv_etas_bulk():
    print('app_support.export_csv_etas_bulk')
    sys.stdout.flush()

def export_csv_etas_single():
    print('app_support.export_csv_etas_single')
    sys.stdout.flush()

def plot_2d_catalog():
    print('app_support.plot_2d_catalog')
    sys.stdout.flush()

def plot_2d_exposure():
    print('app_support.plot_2d_exposure')
    sys.stdout.flush()

def plot_2d_exposure_catalog():
    print('app_support.plot_2d_exposure_catalog')
    sys.stdout.flush()

def plot_3d_catalog():
    print('app_support.plot_3d_catalog')
    sys.stdout.flush()

def plot_3d_etas():
    print('app_support.plot_3d_etas')
    sys.stdout.flush()

def plot_3d_exposure():
    print('app_support.plot_3d_exposure')
    sys.stdout.flush()

def plot_3d_exposure_catalog():
    print('app_support.plot_3d_exposure_catalog')
    sys.stdout.flush()

def plot_animation_etas_bulk():
    print('app_support.plot_animation_etas_bulk')
    sys.stdout.flush()

def plot_animation_etas_single():
    print('app_support.plot_animation_etas_single')
    sys.stdout.flush()

def plot_gmm():
    print('app_support.plot_gmm')
    sys.stdout.flush()

def plot_length_etas_bulk():
    print('app_support.plot_length_etas_bulk')
    sys.stdout.flush()

def plot_mag_freq_distr():
    print('app_support.plot_mag_freq_distr')
    sys.stdout.flush()

def plot_mag_incompl():
    print('app_support.plot_mag_incompl')
    sys.stdout.flush()

def plot_productivity():
    print('app_support.plot_productivity')
    sys.stdout.flush()

def plot_space_etas_bulk():
    print('app_support.plot_space_etas_bulk')
    sys.stdout.flush()

def plot_space_etas_single():
    print('app_support.plot_space_etas_single')
    sys.stdout.flush()

def plot_space_pdf():
    print('app_support.plot_space_pdf')
    sys.stdout.flush()

def plot_time_etas_bulk():
    print('app_support.plot_time_etas_bulk')
    sys.stdout.flush()

def plot_time_etas_single():
    print('app_support.plot_time_etas_single')
    sys.stdout.flush()

def plot_time_pdf():
    print('app_support.plot_time_pdf')
    sys.stdout.flush()

def plot_webmap_catalog():
    print('app_support.plot_webmap_catalog')
    sys.stdout.flush()

def run_etas():
    print('app_support.run_etas')
    sys.stdout.flush()

def upload_csv_catalog():
    print('app_support.upload_csv_catalog')
    sys.stdout.flush()

def upload_csv_depth():
    print('app_support.upload_csv_depth')
    sys.stdout.flush()

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




