# pysimulator
Simulator package for earthquake catalogs with Epidemic-Type Aftershock Sequence (ETAS) - Salvatore Iacoletti

These codes were used to produce the results published in Iacoletti et al. (2022):
- [pyetas](https://github.com/SalvIac/pyetas) was used to calibrate the ETAS models;
- pysimulator was used to simulate synthetic catalogs using the calibrated ETAS models.

Please cite:

Iacoletti S., Cremen G., Galasso C.; Validation of the Epidemic‐Type Aftershock Sequence (ETAS) Models for Simulation‐Based Seismic Hazard Assessments. Seismological Research Letters 2022; doi: https://doi.org/10.1785/0220210134


## Available functionalities
- Simulation starting from single event scenario or full-scale earthquake catalogs;
- Simulation of events taking into account the geometry of the fault ruptures. A few examples of fault geometry are shown in Iacoletti et al. (2022);
- Simulation of various types of ETAS models, especially the one described in Iacoletti et al. (2022), including short-term incompleteness, truncated time and magnitude distributions;
- Flexible: pysimulator is object-oriented for easy implementation of new functionalities in the future;
- Spatially varying background seismicity;
- Simulation of nodal planes and depths.

## Future improvements
- Add more examples to showcase pysimulator capabilities;
- Complete Zhuang-type background seismicity;
- Full hazard/risk calculations (most likely, this will be in a different public repository);
- Graphic User Interface (GUI): in development.
