# Plotting interface to GRASS GIS, based on Basemap

# Starting out with a static projection, but this can be defined by functions in the future

from mpl_toolkits.basemap import Basemap, cm
import numpy as np
import matplotlib.pyplot as plt
from grass import script as grass
from grass.script import array as garray
from matplotlib.colors import Normalize, LinearSegmentedColormap
import re

def read_vector_lines(vect, area=True):
  # First the data
  # Parse the vertices from v.out.ascii
  all_lines_output = []
  vertices_raw = grass.read_command('v.out.ascii', input=vect, output='-', type='line,boundary', format='wkt')
  vector_lines = vertices_raw.split('\n')
  for vector_line in vector_lines:
    if vector_line != '': # Last line should be empty, this will remove it safely
      vertices_output = []
      # strips parentheses and text, and then separates out coordiante pairs
      vertex_list = re.sub("[A-Z]|\(|\)", "", vector_line).split(', ')
      # Turn coordiante pairs into a numpy array and add to the output list
      all_lines_output.append( np.array([vertex.split() for vertex in vertex_list]).astype(float) )
  # And then the other attributes to go along with them
  """
  if area == True:
    attributes_raw = grass.read_command('v.out.ascii', input=vect, output='-', type='point,centroid', format='point')  
    attributes_list = attributes_raw.split('\n')[:-1]
    centroids = np.array( [centroid.split('|') for centroid in attributes_list] )
    attributes = centroids[:,2:]
    categories = attributes[:,0].astype(int)
  """
  return all_lines_output

class grassplot(object):

  def __init__(self, basemap_projection):
    self.m = basemap_projection
    # self.grass_projection = grass.parse_command('g.proj', flags='j').get('+proj')
    # grass.run_command('g.gisenv', set="G_VERBOSE=-1") # Trying to make it quiet!
  
  def rastprep(self, raster_grid_name, resolution=90, figsize=(6,8)):#, colormap=cm.GMT_haxby, alpha=1):
    # handle the flipud and resolution (per above function)
    # also use any set transparency
    # Send input to class-wide variables and set the resolution
    self.raster_grid_name = raster_grid_name
    self.resolution = resolution
    self.figsize = figsize
    self.set_resolution()
    # Then get the grid from GRASS
    self.rast_grid = garray.array()
    self.rast_grid.read(raster_grid_name)
    self.rast_grid = np.flipud(self.rast_grid)
    self.buffer_rast_grid() # put nan's around it and extend n, s, w, e, lats, lons, nlats, nlons, to prevent streaking
    # And transform it into the coordiante system
    rast_grid_transformed = self.m.transform_scalar(self.rast_grid, self.lons, self.lats,self.nlons,self.nlats)
    return rast_grid_transformed
    # Plot
    #fig = plt.figure(figsize=figsize)
    #self.m.imshow(rast_grid_transformed, cmap=colormap, alpha=alpha)

  def buffer_rast_grid(self):
    if self.e + np.diff(self.lons)[-1] < 180:
      self.e += np.diff(self.lons)[-1]
      self.lons = np.concatenate(( self.lons, [self.lons[-1] + np.diff(self.lons)[-1]] ))
      self.rast_grid = np.hstack((self.rast_grid, np.nan*np.zeros((self.rast_grid.shape[0],1)) ))
    if self.w - np.diff(self.lons)[0] > - 180:
      self.w -= np.diff(self.lons)[0]
      self.lons = np.concatenate(( [self.lons[0] - np.diff(self.lons)[0]], self.lons ))
      self.rast_grid = np.hstack((np.nan*np.zeros((self.rast_grid.shape[0],1)), self.rast_grid ))
    if self.s + np.diff(self.lats)[0] > -90:
      self.s -= np.diff(self.lats)[0]
      self.lats = np.concatenate(( [self.lats[0] - np.diff(self.lats)[0]], self.lats ))
      self.rast_grid = np.vstack((self.rast_grid, np.nan*np.zeros((1,self.rast_grid.shape[1])) ))
    if self.n + np.diff(self.lats)[-1] < 90:
      self.n += np.diff(self.lats)[-1]
      self.lats = np.concatenate(( self.lats, [self.lats[-1] + np.diff(self.lats)[-1]] ))
      self.rast_grid = np.vstack((np.nan*np.zeros((1, self.rast_grid.shape[1])), self.rast_grid))
    
  def set_resolution(self):
    """
    resolution is in dpi, so is a function of figsize
    """
    # Get maximum resolution
    #raster_region = self.parse_region( grass.read_command('g.region', rast=self.raster_grid_name, flags='p') )#, flags='up') ) "u" doesn't change the region, so "up" just prints it out
    raster_region = self.parse_region( grass.read_command('g.region', n=85, s=15, w=-170, e=-40, res='0:0:30', flags='p') )#, flags='up') ) "u" doesn't change the region, so "up" just prints it out
    rast_nlats = float(raster_region['rows'])
    rast_nlons = float(raster_region['cols'])
    self.nlats = int(np.min((rast_nlats, self.figsize[0]*self.resolution)))
    self.nlons = int(np.min((rast_nlons, self.figsize[1]*self.resolution)))
    grass.run_command('g.region', rows=self.nlats, cols=self.nlons)
    self.s = grass.region()['s']
    self.n = grass.region()['n']
    self.w = grass.region()['w']
    self.e = grass.region()['e']
    # And also set the lats and lons for the Basemap grid
    # use np.mean to get the cell centers
    self.lats = self.midpoints( np.linspace(self.s, self.n, self.nlats+1) )
    self.lons = self.midpoints( np.linspace(self.w, self.e, self.nlons+1) )
  
  def midpoints(self, invar):
    return (invar[1:] + invar[:-1]) / 2
  
  def parse_region(self, grassoutput):
    prepped = re.sub(': +','\t',grassoutput)
    output = prepped.split('\n')
    for i in range(len(output)):
      if output[i] == '':
        output.pop(i)
      else:
        output[i] = output[i].split('\t')
    return dict(output)
  
  """
  def project(self, projection = self.grass_projection):
    # Just pass m to this function: created by script calling this class
    self.m = Basemap(projection='stere', lon_0=-98., lat_0=90., lat_ts=90.,\
            llcrnrlat=23,urcrnrlat=55,\
            llcrnrlon=-117,urcrnrlon=-45,\
            rsphere=6371200.,resolution='l',area_thresh=10000)
            
    nx = grass.region()['cols']
    ny = grass.region()['rows']
    
  # transform to nx x ny regularly spaced 5km native projection grid
  nx = int((m.xmax-m.xmin)/5000.)+1; ny = int((m.ymax-m.ymin)/5000.)+1
  topodat = m.transform_scalar(topoin,lons,lats,nx,ny)
  # plot image over map with imshow.
  im = m.imshow(topodat,cm.GMT_haxby)
  """

  def plot_figure(self, figure_width, figure_height):
    self.fig = plt.figure( figsize=(figure_width, figure_height) )
    
  def make_GRASS_etopo2_colormap(self):
    """
    GRASS GIS allows for color maps to be assigned to absolute values.
    Matplotlib doesn't seem to.
    So this will import and interpolate the etopo2 color map.
    
    """
    etopo2 = np.genfromtxt('GRASScolors/etopo2', skip_footer=1)
    z = etopo2[:,0].astype(int)
    r = etopo2[:,1].astype(float)
    g = etopo2[:,2].astype(float)
    b = etopo2[:,3].astype(float)
    from scipy.interpolate import interp1d
    ri = interp1d(z, r)
    gi = interp1d(z, g)
    bi = interp1d(z, b)
    low_elev = np.min(z)
    high_elev = np.max(z)
    znew = np.linspace(low_elev, high_elev, 512)
    znew = np.concatenate(( znew[znew<-1], [-1, 0], znew[znew>0])) # make sure key SL transition is intact!
    rnew = ri(znew)
    gnew = gi(znew)
    bnew = bi(znew)
    clscaled = np.linspace(0, 1, len(znew))
    cdr = []
    cdg = []
    cdb = []
    for i in range(len(znew)):
      cdr.append([clscaled[i], rnew[i]/255., rnew[i]/255.])
      cdg.append([clscaled[i], gnew[i]/255., gnew[i]/255.])
      cdb.append([clscaled[i], bnew[i]/255., bnew[i]/255.])
    cdict = {'red': cdr, 'green': cdg, 'blue': cdb}
    cm_etopo2 = LinearSegmentedColormap('etopo2',cdict,4096)
    return low_elev, high_elev, cm_etopo2
    
    
  def contour(self, filled=False):
    # Build grids with the x and y data and then make contour raster map
    pass
    
  def get_nice_colormap(self):
    pass
    # Per Ethan's suggestion
    
  def text_label(self):
    pass
  
  def export_to_svg(self):
    pass
    # or just use savefig with svg format specified?
    
# Colorbar is bipolar:
# http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

class colorbar_bipolar(Normalize):    
  def __init__(self,linthresh,vmin=None,vmax=None,clip=False):
    Normalize.__init__(self,vmin,vmax,clip)
    self.linthresh=linthresh
    self.vmin, self.vmax = vmin, vmax
    
  def __call__(self, value, clip=None):
    if clip is None:
      clip = self.clip

    result, is_scalar = self.process_value(value)

    self.autoscale_None(result)
    vmin, vmax = self.vmin, self.vmax
    if vmin > 0:
      raise ValueError("minvalue must be less than 0")
    if vmax < 0:
      raise ValueError("maxvalue must be more than 0")            
    elif vmin == vmax:
      result.fill(0) # Or should it be all masked? Or 0.5?
    else:
      vmin = float(vmin)
      vmax = float(vmax)
      if clip:
        mask = ma.getmask(result)
        result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                          mask=mask)
      # ma division is very slow; we can take a shortcut
      resdat = result.data

      resdat[resdat>0] /= vmax
      resdat[resdat<0] /= -vmin
      resdat=resdat/2.+0.5
      result = np.ma.array(resdat, mask=result.mask, copy=False)

    if is_scalar:
        result = result[0]

    return result

  def inverse(self, value):
    if not self.scaled():
      raise ValueError("Not invertible until scaled")
    vmin, vmax = self.vmin, self.vmax

    if cbook.iterable(value):
      val = ma.asarray(value)
      val=2*(val-0.5) 
      val[val>0]*=vmax
      val[val<0]*=-vmin
      return val
    else:
      if val<0.5: 
        return  2*val*(-vmin)
      else:
        return val*vmax

  def makeTickLabels(self, nlabels):
    #proportion_min = np.abs(self.vmin - self.linthresh) / ( np.abs(self.vmin - self.linthresh) + np.abs(self.vmax - self.linthresh) )
    #nlabels_min = np.round((nlabels - 1) * proportion_min) # save the last label for the midpoint
    #nlabels_max = nlabels - 1 - nlabels_min
    # Will always add a point at the middle
    ticks = np.concatenate(( np.linspace(0, 0.5, nlabels/2+1), np.linspace(.5, 1, nlabels/2+1)[1:]))
    tick_labels = np.concatenate(( np.linspace(self.vmin, self.linthresh, nlabels/2 + 1), np.linspace(self.linthresh, self.vmax, nlabels/2 + 1)[1:] ))
    tick_labels = list(tick_labels)
    for i in range(len(tick_labels)):
      tick_labels[i] = '%.2f' %tick_labels[i]
    return ticks, tick_labels

