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
from openquake.hazardlib.geo.surface.complex_fault import ComplexFaultSurface
from openquake.hazardlib.geo.surface.multi import MultiSurface
from openquake.hazardlib.source.rupture import (
                                            ParametricProbabilisticRupture,
                                            NonParametricProbabilisticRupture,
                                            PointRupture)
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.const import TRT
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.line import Line
from openquake.hazardlib.scalerel import PointMSR
from openquake.hazardlib.geo.nodalplane import NodalPlane

from pysimulator.build_rupture_plane import get_rupture_surface
from pysimulator.build_point_plane import get_point_surface

'''
RuptureBuilder can build a rupture (ParametricProbabilisticRupture in OQ) with
fictitious rate (it should not be used) in the following cases:
    - simple (point source, init_point): mag, lon, lat, depth, strike, dip, rake
    - medium (init_surface_from_point): mag, lon, lat, depth,
                                        strike, dip, rake, mesh_spacing=1.
    - complex (init_surface): mag, lon, lat, depth,
                              rake, surface (OQ surface object)
'''


class RuptureBuilder():

    @classmethod
    def init_surface(cls, mag, lon, lat, depth, rake, surface):
        tectonic_region_type = TRT.ACTIVE_SHALLOW_CRUST #TODO
        hypocenter = Point(lon, lat, depth)
        occurrence_rate = 1.
        temporal_occurrence_model = PoissonTOM
        rupture = ParametricProbabilisticRupture(mag, rake, tectonic_region_type, 
                                                 hypocenter, surface,
                                                 occurrence_rate,
                                                 temporal_occurrence_model)
        return rupture
    
    @classmethod
    def init_point(cls, mag, lon, lat, depth, rake=None, strike=None, dip=None):
        tectonic_region_type = TRT.ACTIVE_SHALLOW_CRUST #TODO
        if lon > 180:
            lon = lon - 360
        hypocenter = Point(lon, lat, depth)
        occurrence_rate = 1.
        temporal_occurrence_model = PoissonTOM
        if (strike is None) or (dip is None) or (strike is None):
            # fictitious
            rake = 0.0
            strike=0.0
            dip=90.0
        rupture = PointRupture(mag, tectonic_region_type, hypocenter,
                               strike, dip, rake, occurrence_rate,
                               temporal_occurrence_model)
        return rupture

    @classmethod
    def init_surface_from_point(cls, mag, lon, lat, depth, strike, dip, rake,
                                mesh_spacing=1.):
        if (strike is None) or (dip is None) or (strike is None):
            # create fictitious (really small) surface
            rake = 0.0
            nodal_plane = NodalPlane(strike=0.0, dip=90.0, rake=rake)
            hypocenter = Point(lon, lat, depth)
            msr = PointMSR()
            surface, hypo = get_point_surface(mag, msr, nodal_plane, hypocenter)
        else:
            lines = cls.get_rupture_lines(mag, lon, lat, depth, strike, dip, rake)
            surface = cls.get_complex_surface(lines, mesh_spacing)
        return cls.init_surface(mag, lon, lat, depth, rake, surface)
    
    @classmethod
    def get_rupture_lines(cls, mag, lon, lat, depth, strike, dip, rake):
        hypocenter = {"lon": lon, "lat": lat, "depth": depth}
        surface_dict = get_rupture_surface(mag, hypocenter, strike, dip, rake)
        lines = cls.dict2lines(surface_dict)
        return lines

    @classmethod
    def get_complex_surface(cls, lines, mesh_spacing):
        return ComplexFaultSurface.from_fault_data(lines, mesh_spacing=mesh_spacing)
    # from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface
    # surface = SimpleFaultSurface.from_fault_data(line_top, 0., 30., 45., 0.1)
    # from openquake.hazardlib.geo.surface.planar import PlanarSurface
    # PlanarSurface.from_corner_points(top_left, top_right, bottom_right, bottom_left)
    # MultiSurface(surfaces, tol=0.1)

    @classmethod
    def dict2lines(cls, surf_dict):
        surf_dict["topLeft"]["lon"]
        lons = [ surf_dict["topLeft"]["lon"], surf_dict["topRight"]["lon"] ]
        lats = [ surf_dict["topLeft"]["lat"], surf_dict["topRight"]["lat"] ]
        depths = [ surf_dict["topLeft"]["depth"], surf_dict["topRight"]["depth"] ]
        line_top = cls.get_line(lons, lats, depths)
        lons = [ surf_dict["bottomLeft"]["lon"], surf_dict["bottomRight"]["lon"] ]
        lats = [ surf_dict["bottomLeft"]["lat"], surf_dict["bottomRight"]["lat"] ]
        depths = [ surf_dict["bottomLeft"]["depth"], surf_dict["bottomRight"]["depth"] ]
        line_bot = cls.get_line(lons, lats, depths)
        return [line_top, line_bot]
    
    @classmethod
    def get_line(cls, lons, lats, depths):
        points = list()
        for lon, lat, dep in zip(lons, lats, depths):
            p = Point(lon, lat, dep)
            points.append(p)
            line = Line(points)
        return line
    
    @classmethod
    def random_hypocenter(cls, surface):
        if surface.__class__.__name__ == "MultiSurface":
            surface = np.random.choice(surface.surfaces)
        if len(surface.mesh.lons.shape) == 2:
            lons = surface.mesh.lons[1:-1,1:-1].flatten() # avoid edges
            lats = surface.mesh.lats[1:-1,1:-1].flatten()
            depths = surface.mesh.depths[1:-1,1:-1].flatten()
        else:
            lons = surface.mesh.lons
            lats = surface.mesh.lats
            depths = surface.mesh.depths
        ind = np.random.choice(list(range(0, lons.shape[0])))
        lon = lons[ind]
        lat = lats[ind]
        depth = depths[ind]
        # # check
        # from pyutils.plot_rup_simple import PlotRup
        # pr = PlotRup()
        # pr.plot_mesh(surface.mesh, s=1)
        # pr.plot_xyz(lon, lat, depth, marker="*", s=200)
        return lon, lat, depth


    @classmethod
    def surface_change_mesh(cls, surface, dim=1.):
        if surface.__class__.__name__ == "MultiSurface":
            surfaces = surface.surfaces
        else:
            surfaces = [surface]
        newsurfs = list()
        for surf in surfaces:
            if surf.mesh.__class__.__name__ == "RectangularMesh":
                lines = list()
                for i in [0, surf.mesh.lons.shape[0]-1]: # doing every line here is slow
                    lons = surf.mesh.lons[i,:]
                    lats = surf.mesh.lats[i,:]
                    depths = surf.mesh.depths[i,:]
                    lines.append( RuptureBuilder.get_line(lons, lats, depths) )
                newsurfs.append( RuptureBuilder.get_complex_surface(lines, dim) )
            else:
                raise Exception("not RectangularMesh")
        if surface.__class__.__name__ == "MultiSurface":
            newsurf = MultiSurface(newsurfs)
        else:
            newsurf = newsurfs[0]
        # # check
        # from pyutils.plot_rup_simple import PlotRup
        # pr = PlotRup()
        # for surf in surface.surfaces:
        #     pr.plot_mesh(surf.mesh, s=1, c="b")
        # for surf in newsurf.surfaces:
        #     pr.plot_mesh(surf.mesh, s=10, c="r")
        # pr.show()
        return newsurf
    
    
    
    