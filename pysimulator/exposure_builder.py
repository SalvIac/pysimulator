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

import tempfile
import datetime
import warnings
import numpy as np
from openquake.risklib.asset import Exposure


'''
ExposureBuilder can build an Exposure model (i.e., a portofolio) in the
following cases:

    - single point (from_point): lon, lat, taxonomy, number=1., filename=None
    - grid (from_inputs): lon, lat, lon_width, lat_height, 
                          lon_bin, lat_bin, taxonomy, number=1., filename=None
    - square grid in lat and lon direction (from_inputs_simm): lon, lat, dim, bin,
                                            taxonomy, number=1., filename=None
    - advanced, from openquake model (from_file): filename
'''


class ExposureBuilder():

    
    @classmethod
    def check_taxonomy(cls, taxonomy):
        # sanity check on taxonomy
        if not np.equal(1., sum([taxonomy[tax] for tax in taxonomy.keys()])):
            raise Exception("taxonomy percentages must sum up to 1.")


    @classmethod
    def from_point(cls, lon, lat, taxonomy, number=1., filename=None):
        '''
        lon lat: center of the portfolio
        number: number of building associated with each point (regardless the taxonomy)
        taxonomy: dict e.g., {"RC": 0.57, "bla": 0.43}
        '''
        cls.check_taxonomy(taxonomy)
        assets = ""
        i_d = 0
        assets, i_d = cls.for_taxonomy(assets, i_d, taxonomy,
                                       number, lon, lat)
        exp = cls.get_exposure(assets[:-1])
        filename = cls.save_xml(exp, filename)
        return cls.from_file(filename)


    @classmethod
    def for_taxonomy(cls, assets, i_d, taxonomy, number, lon, lat):
        for tax in taxonomy.keys():
            i_d += 1
            num = taxonomy[tax] * number
            assets += cls.get_asset(i_d, lon, lat, tax, num) + "\n"
        return assets, i_d



    @classmethod
    def from_points(cls, lons, lats, taxonomies, number=1., filename=None):
        '''
        lons lats: locations
        number: number of building associated with each point (regardless the taxonomy)
        taxonomies: dict e.g., {"RC": 0.57, "bla": 0.43} or list of dict
        '''
        assets = ""
        i_d = 0
        if isinstance(taxonomies, dict):
            taxonomy = taxonomies
            cls.check_taxonomy(taxonomy)
            for lon, lat in zip(lons, lats):
                assets, i_d = cls.for_taxonomy(assets, i_d, taxonomy,
                                               number, lon, lat)
        elif isinstance(taxonomies, list):
            for lon, lat, taxonomy in zip(lons, lats, taxonomies):
                cls.check_taxonomy(taxonomy)
                assets, i_d = cls.for_taxonomy(assets, i_d, taxonomy,
                                               number, lon, lat)
        exp = cls.get_exposure(assets[:-1])
        filename = cls.save_xml(exp, filename)
        return cls.from_file(filename)
    

    @classmethod
    def from_inputs_square(cls, lon, lat, dim, bin,
                         taxonomy, number=1., value=1., filename=None):
        return cls.from_inputs(lon, lat, dim, dim, bin, bin,
                               taxonomy, number, "Rectangle", filename)


    @classmethod
    def from_inputs(cls, lon_center, lat_center, lon_width, lat_height,
                    lon_bin, lat_bin, taxonomy, number=1.,
                    geometry="Rectangle", filename=None):
        '''
        lon lat: center of the portfolio
        lon_width, lat_height: dimensions of the rectangular grid portfolio
        lon_bin, lat_bin: create grid every "bin" for lon and lat
        number: number of building associated with each point (regardless the taxonomy)
        value: monetary value of assets
        geometry: it can be "Rectangle" or "Circle""
        taxonomy: dict e.g., {"RC": 0.57, "bla": 0.43}
        '''
        cls.check_taxonomy(taxonomy)
        mesh_lon, mesh_lat = cls.get_mesh_port(lon_center, lat_center,
                                               lon_width, lat_height,
                                               lon_bin, lat_bin, geometry)
        assets = ""
        i_d = 0
        for lon, lat in zip(mesh_lon, mesh_lat):
            for tax in taxonomy.keys():
                i_d += 1
                num = taxonomy[tax] * number
                assets += cls.get_asset(i_d, lon, lat, tax, num) + "\n"
        exp = cls.get_exposure(assets[:-1])
        filename = cls.save_xml(exp, filename)
        return cls.from_file(filename)
    
    
    @classmethod
    def get_mesh_port(cls, lon, lat, lon_width, lat_height, 
                      lon_bin, lat_bin, geometry):
        lon_r = lon+lon_width/2
        lon_l = lon-lon_width/2
        lat_t = lat+lat_height/2
        lat_b = lat-lat_height/2
        
        lon_range = np.arange(lon_l, lon_r+lon_bin, lon_bin)
        lat_range = np.arange(lat_b, lat_t+lat_bin, lat_bin)
        mesh = np.meshgrid(lon_range, lat_range)
        mesh_lon = mesh[0].flatten()
        mesh_lat = mesh[1].flatten()
        if geometry == "Circle":
            r = min(lon_width/2, lat_height/2)+1e-6
            dists = np.sqrt((mesh_lon-lon)**2+(mesh_lat-lat)**2)
            mesh_lon = mesh_lon[dists <= r]
            mesh_lat = mesh_lat[dists <= r]
        # check if lon lat is in mesh, otherwise warning
        check = np.any(np.logical_and(np.isclose(lon, mesh_lon),
                                      np.isclose(lat, mesh_lat)))
        if not check:
            warnings.warn("Warning: center of portolio not in final")
        return mesh_lon, mesh_lat
    
    
    @classmethod
    def from_file(cls, filename):
        return Exposure.read([filename])


    @classmethod
    def save_xml(cls, xml_text, filename=None):
        if filename is None:
            fh, filename = tempfile.mkstemp(prefix="exp", suffix=".xml")
        with open(filename, "w") as f:
           f.write(xml_text)
        return filename
           

    @classmethod
    def get_assets(cls,  lons, lats, taxonomies, numbers):
        assets = ""
        for i_d, (num, tax, lon, lat) in enumerate(zip(numbers, taxonomies, lons, lats)):
            assets = assets + cls.get_asset(i_d, lon, lat, tax, num) + "\n"
        return assets[:-1]
    
    
    @classmethod
    def get_asset(cls, i_d, lon, lat, taxonomy, number):
        return '''			<asset id="{}" number="{}"  taxonomy="{}" >
				<location lon="{}" lat="{}" />
				<costs>
				</costs>
			</asset>'''.format(i_d, number, taxonomy, lon, lat)
    
    
    @classmethod
    def get_exposure(cls, assets):
        return '''<?xml version="1.0" encoding="UTF-8"?>
<nrml xmlns="http://openquake.org/xmlns/nrml/0.4">
	<exposureModel id="ex1" category="buildings" taxonomySource="">
		<description>Automatically Generated {}</description>
		<conversions>
			<costTypes>
			</costTypes>
		</conversions>
		<assets>
      {}
		</assets>
	</exposureModel>
</nrml>'''.format(str(datetime.datetime.now()), assets)
    


#%%

if __name__ == "__main__":

    # bla = ExposureBuilder.get_assets([10,20], [20,10], ["RC", "RC"], [1,2])
    # bla = ExposureBuilder.get_exposure(bla)
    # bla2 = ExposureBuilder.save_xml(bla)
    # bla3 = ExposureBuilder.from_file(bla2)

    lon = 174.75
    lat = -41.25
    lon_width = 0.5
    lat_height = 0.6
    lon_bin, lat_bin = 0.01, 0.02
    taxonomy = {"RC1": 0.57, "RC2": 0.43}
    exp1 = ExposureBuilder.from_inputs(lon, lat,
                                lon_width, lat_height, 
                                lon_bin, lat_bin,
                                taxonomy, number=1., geometry="Rectangle")
    
    exp1c = ExposureBuilder.from_inputs(lon, lat,
                                lon_width, lat_height, 
                                lon_bin, lat_bin,
                                taxonomy, number=1., geometry="Circle")    
    
    exp2 = ExposureBuilder.from_inputs_square(lon, lat, 0.5, 0.01,
                                            taxonomy, number=1.)
    
    exp3 = ExposureBuilder.from_point(lon, lat,
                                      taxonomy, number=1.)
    
    lons = [174.75, 174.5]
    lats = [-41.5, -41.2]
    taxonomies = [taxonomy, taxonomy]
    exp4 = ExposureBuilder.from_points(lons, lats,
                                       taxonomies, number=1.)

    