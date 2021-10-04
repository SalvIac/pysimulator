# -*- coding: utf-8 -*-
"""
@author: Salvatore
"""

# inspired by OQ point source class



import math
import numpy as np
from openquake.hazardlib.scalerel import PointMSR
from openquake.hazardlib.geo import Point
from openquake.hazardlib.geo.surface.complex_fault import ComplexFaultSurface
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.geo.mesh import RectangularMesh





def get_point_surface(mag, msr, nodal_plane, hypocenter):
    """
    Create and return rupture surface object with given properties.
    This is from a point "source" (I know the name of the function is ambiguous)
    """
    upper_seismogenic_depth = hypocenter.depth-1
    lower_seismogenic_depth = hypocenter.depth+1
    eps = .001  # 1 meter buffer to survive numerical errors
    assert upper_seismogenic_depth < hypocenter.depth + eps, (
        upper_seismogenic_depth, hypocenter.depth)
    assert lower_seismogenic_depth + eps > hypocenter.depth, (
        lower_seismogenic_depth, hypocenter.depth)
    rdip = math.radians(nodal_plane.dip)

    # precalculated azimuth values for horizontal-only and vertical-only
    # moves from one point to another on the plane defined by strike
    # and dip:
    azimuth_right = nodal_plane.strike
    azimuth_down = (azimuth_right + 90) % 360
    azimuth_left = (azimuth_down + 90) % 360
    azimuth_up = (azimuth_left + 90) % 360

    rup_length, rup_width = _get_rupture_dimensions(
                        msr, mag, nodal_plane.rake, nodal_plane.dip,
                        1., upper_seismogenic_depth, lower_seismogenic_depth)
    # calculate the height of the rupture being projected
    # on the vertical plane:
    rup_proj_height = rup_width * math.sin(rdip)
    # and it's width being projected on the horizontal one:
    rup_proj_width = rup_width * math.cos(rdip)

    # half height of the vertical component of rupture width
    # is the vertical distance between the rupture geometrical
    # center and it's upper and lower borders:
    hheight = rup_proj_height / 2.
    # calculate how much shallower the upper border of the rupture
    # is than the upper seismogenic depth:
    vshift = upper_seismogenic_depth - hypocenter.depth + hheight
    # if it is shallower (vshift > 0) than we need to move the rupture
    # by that value vertically.
    if vshift < 0:
        # the top edge is below upper seismogenic depth. now we need
        # to check that we do not cross the lower border.
        vshift = lower_seismogenic_depth - hypocenter.depth - hheight
        if vshift > 0:
            # the bottom edge of the rupture is above the lower sesmogenic
            # depth. that means that we don't need to move the rupture
            # as it fits inside seismogenic layer.
            vshift = 0
        # if vshift < 0 than we need to move the rupture up by that value.

    # now we need to find the position of rupture's geometrical center.
    # in any case the hypocenter point must lie on the surface, however
    # the rupture center might be off (below or above) along the dip.
    rupture_center = hypocenter
    if vshift != 0:
        # we need to move the rupture center to make the rupture fit
        # inside the seismogenic layer.
        hshift = abs(vshift / math.tan(rdip))
        rupture_center = rupture_center.point_at(
            horizontal_distance=hshift, vertical_increment=vshift,
            azimuth=(azimuth_up if vshift < 0 else azimuth_down))

    # from the rupture center we can now compute the coordinates of the
    # four coorners by moving along the diagonals of the plane. This seems
    # to be better then moving along the perimeter, because in this case
    # errors are accumulated that induce distorsions in the shape with
    # consequent raise of exceptions when creating PlanarSurface objects
    # theta is the angle between the diagonal of the surface projection
    # and the line passing through the rupture center and parallel to the
    # top and bottom edges. Theta is zero for vertical ruptures (because
    # rup_proj_width is zero)
    theta = math.degrees(
        math.atan((rup_proj_width / 2.) / (rup_length / 2.)))
    hor_dist = math.sqrt(
        (rup_length / 2.) ** 2 + (rup_proj_width / 2.) ** 2)

    left_top = rupture_center.point_at(
        horizontal_distance=hor_dist,
        vertical_increment=-rup_proj_height / 2.,
        azimuth=(nodal_plane.strike + 180 + theta) % 360)
    right_top = rupture_center.point_at(
        horizontal_distance=hor_dist,
        vertical_increment=-rup_proj_height / 2.,
        azimuth=(nodal_plane.strike - theta) % 360)
    left_bottom = rupture_center.point_at(
        horizontal_distance=hor_dist,
        vertical_increment=rup_proj_height / 2.,
        azimuth=(nodal_plane.strike + 180 - theta) % 360)
    right_bottom = rupture_center.point_at(
        horizontal_distance=hor_dist,
        vertical_increment=rup_proj_height / 2.,
        azimuth=(nodal_plane.strike + theta) % 360)
    # surface = PlanarSurface(
    #     nodal_plane.strike, nodal_plane.dip, left_top, right_top,
    #     right_bottom, left_bottom)
    mesh = RectangularMesh(np.array([[left_top.longitude, right_top.longitude],
                                     [left_bottom.longitude, right_bottom.longitude]]),
                           np.array([[left_top.latitude, right_top.latitude],
                                     [left_bottom.latitude, right_bottom.latitude]]),
                           np.array([[left_top.depth, right_top.depth],
                                     [left_bottom.depth, right_bottom.depth]]))
    surface = ComplexFaultSurface(mesh)
    return surface, rupture_center



def _get_rupture_dimensions(magscalrel, mag, rake, dip, rupture_aspect_ratio, 
                            upper_seismogenic_depth, lower_seismogenic_depth):
    """
    Calculate and return the rupture length and width
    for given magnitude ``mag`` and nodal plane.
    """
    area = magscalrel.get_median_area(mag, rake)
    rup_length = math.sqrt(area * rupture_aspect_ratio)
    rup_width = area / rup_length
    seismogenic_layer_width = (lower_seismogenic_depth
                               - upper_seismogenic_depth)
    max_width = seismogenic_layer_width / math.sin(math.radians(dip))
    if rup_width > max_width:
        rup_width = max_width
        rup_length = area / rup_width
    return rup_length, rup_width



#%%

if __name__ == "__main__":
    nodal_plane = NodalPlane(strike=0.0, dip=90.0, rake=0.0)
    hypocenter = Point(latitude=-40.200000, longitude=174.000000, depth=10.0000)
    mag = 7. # the result must not change with the mag
    msr = PointMSR()
    surface, hypo = get_point_surface(mag, msr, nodal_plane, hypocenter)
    vars(surface)
    
