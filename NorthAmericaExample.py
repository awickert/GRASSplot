#! /usr/bin/python
# ADW, 13 October 2013
# To use this example, you will have to create two maps of North America, 
# "topo" and "shaded". I suggest that you import etopo1 and then use
# r.shaded.relief.
# This only works for locations with lat/lon (unprojected) coordinates

import grassplot as gp
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import pyplot as plt
import numpy as np
from grass import script as grass

m = Basemap(width=7000000,height=6500000,
            resolution='l',projection='laea',\
            lat_ts=52,lat_0=52,lon_0=-100.)

p = gp.grassplot(m)
etopo2low, etopo2high, etopo2cm = p.make_GRASS_etopo2_colormap()

axsize = (9., 10.)
plt.figure(figsize=axsize)

# RASTER
# Prep
# Shaded relief example
# Can add more parameters to this to select resolution of map display, etc.
# (changing resolution is how this can be not-too-slow even for large rasters)
# resolution is in dpi
shaded = p.rastprep('shaded', figsize=axsize, resolution=90) 
topo = p.rastprep('topo', figsize=axsize, resolution=90)
# Plotting
shademap = m.imshow(shaded, cmap=plt.cm.Greys)
topomap = m.imshow(topo, cmap=etopo2cm, vmin=etopo2low, vmax=etopo2high, alpha=.6)

# VECTOR
# For vector files, use:
# gp.read_vector_lines('grassVectorName')

# Matplotlib functionality - let's contour topography just to show what it looks like
# anyway, will add this at some point, just see:
# http://matplotlib.org/basemap/users/examples.html
# for an example

plt.show()

