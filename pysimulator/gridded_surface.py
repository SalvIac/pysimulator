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

from openquake.hazardlib.geo.surface.gridded import GriddedSurface


class GriddedSurface2(GriddedSurface):

    def get_rx_distance(self, mesh):
        """
        Compute distance between each point of mesh and surface's great circle
        arc.

        Distance is measured perpendicular to the rupture strike, from
        the surface projection of the updip edge of the rupture, with
        the down dip direction being positive (this distance is usually
        called ``Rx``).

        In other words, is the horizontal distance to top edge of rupture
        measured perpendicular to the strike. Values on the hanging wall
        are positive, values on the footwall are negative.

        :param mesh:
            :class:`~openquake.hazardlib.geo.mesh.Mesh` of points to calculate
            Rx-distance to.
        :returns:
            Numpy array of distances in km.
        """
        raise NotImplementedError('GriddedSurface')

    def get_top_edge_depth(self):
        """
        Compute minimum depth of surface's top edge.

        :returns:
            Float value, the vertical distance between the earth surface
            and the shallowest point in surface's top edge in km.
        """
        raise NotImplementedError('GriddedSurface')

    def get_strike(self):
        """
        Compute surface's strike as decimal degrees in a range ``[0, 360)``.

        The actual definition of the strike might depend on surface geometry.

        :returns:
            numpy.nan, not available for this kind of surface (yet)
        """
        return np.nan

    def get_dip(self):
        """
        Compute surface's dip as decimal degrees in a range ``(0, 90]``.

        The actual definition of the dip might depend on surface geometry.

        :returns:
            numpy.nan, not available for this kind of surface (yet)
        """
        return np.nan

    def get_width(self):
        """
        Compute surface's width (that is surface extension along the
        dip direction) in km.

        The actual definition depends on the type of surface geometry.

        :returns:
            Float value, the surface width
        """
        raise NotImplementedError('GriddedSurface')

    def get_area(self):
        """
        Compute surface's area in squared km.

        :returns:
            Float value, the surface area
        """
        raise NotImplementedError('GriddedSurface')

    def get_ry0_distance(self, mesh):
        """
        :param mesh:
            :class:`~openquake.hazardlib.geo.mesh.Mesh` of points
        """
        raise NotImplementedError('GriddedSurface')
        
        