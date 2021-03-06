# -*- coding: utf-8 -*-
# pysimulator
#  Copyright (C) 2021-2022 Salvatore Iacoletti
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

from setuptools import setup, find_packages


setup(
    name='pysimulator',
    version='0.0.2',
    description='Simulator package for earthquake catalogs - Salvatore Iacoletti',
    author='Salvatore Iacoletti',
    author_email='salvatore.iacoletti92@gmail.com',
    url='https://github.com/SalvIac/pysimulator',
    packages=find_packages(),
    zip_safe=False,
    )