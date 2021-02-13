#!/usr/bin/env python

import os.path
import sys
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  

#import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ColorConverter

from matplotlib import pylab
#from pylab import colorbar, clf, axes, linspace, pcolor, meshgrid, show, axis, title
#from scitools.easyviz.matplotlib_ import colorbar, clf, axes, linspace,\
             #pcolor, meshgrid, show, colormap

from numpy import sort
try:
    import matplotlib.pyplot as plt
except:
    print("Error importing pyplot from matplotlib, please install matplotlib package first...")
    sys.tracebacklimit=0
    raise Exception("Importing matplotlib failed")

def grid(x, y, z, resX=500, resY=500):
    """
    Converts 3 column data to matplotlib grid
    """
    from matplotlib.mlab import griddata
    #from scipy.interpolate import griddata

    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)

    # mlab version
    Z = griddata(x, y, z, xi, yi, interp='linear')
    # scipy version
    #Z = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

def make_colormap(colors):
    """
    Define a new color map based on values specified in the dictionary
    colors, where colors[z] is the color that value z should be mapped to,
    with linear interpolation between the given values of z.

    The z values (dictionary keys) are real numbers and the values
    colors[z] can be either an RGB list, e.g. [1,0,0] for red, or an
    html hex string, e.g. "#ff0000" for red.
    """
    z = sort(colors.keys())

    n = len(z)
    z1 = min(z)
    zn = max(z)
    x0 = (z - z1) / (zn - z1)

    CC = ColorConverter()
    R = []
    G = []
    B = []
    for i in range(n):
        #i'th color at level z[i]:
        Ci = colors[z[i]]
        if type(Ci) == str:
            # a hex string of form '#ff0000' for example (for red)
            RGB = CC.to_rgb(Ci)
        else:
            # assume it's an RGB triple already:
            RGB = Ci
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])

    cmap_dict = {}
    cmap_dict['red'] = [(x0[i],R[i],R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(x0[i],G[i],G[i]) for i in range(len(G))]
    cmap_dict['blue'] = [(x0[i],B[i],B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap',cmap_dict)
    return mymap

def plot_kernels(filename,show=False):
    """
    plots ASCII kernel file
    """
    print "plotting kernel file: ",filename
    print ""

    data = np.loadtxt(filename)

    # checks data
    if data.ndim != 2:
        print "Error: wrong data dimension for kernel file",data.ndim
        sys.tracebacklimit=0
        raise Exception("Invalid data dimension")

    # checks array
    if len(data[1,:]) != 5:
        print "data shape  : ",data.shape
        print "data lengths: ",len(data[:,1]),len(data[1,:])
        print "Error: wrong data format for kernel file",data.shape
        sys.tracebacklimit=0
        raise Exception("Invalid data format")

    # splits up data
    x = data[:,0]
    y = data[:,1]

    print "dimensions:"
    print "  x-range min/max = %f / %f" % (x.min(), x.max())
    print "  y-range min/max = %f / %f" % (y.min(), y.max())
    print ""
    #ytmp = y.max()
    #y = y.max() - y
    #x = x / x.max() * 9000
    #x = x.max() - x
    z1 = data[:,2] # e.g. rho
    z2 = data[:,3] # e.g. alpha
    z3 = data[:,4] # e.g. beta

    # names like
    #   rhop_alpha_beta_kernel.dat
    # or
    #   proc000000_rhop_alpha_beta_kernel.dat
    name = os.path.basename(file)

    name_kernels = str.split(name,"_")
    if len(name_kernels) == 4:
        kernel1 = 'K_' + name_kernels[0] # rhop
        kernel2 = 'K_' + name_kernels[1] # alpha
        kernel3 = 'K_' + name_kernels[2] # beta
    elif len(name_kernels) == 5:
        kernel1 = 'K_' + name_kernels[1]
        kernel2 = 'K_' + name_kernels[2]
        kernel3 = 'K_' + name_kernels[3]
    else:
        kernel1 = 'K_1'
        kernel2 = 'K_2'
        kernel3 = 'K_3'

    kernel2 = 'VP'
    kernel3 = 'VS'	

    print "statistics:"
    print "  %12s : min/max = %e / %e" % (kernel1,z1.min(),z1.max())
    print "  %12s : min/max = %e / %e" % (kernel2,z2.min(),z2.max())
    print "  %12s : min/max = %e / %e" % (kernel3,z3.min(),z3.max())
    print ""

    total_max = (np.concatenate((z2,z3))).max()*0.9
    #total_min = (np.concatenate((z2,z3))).min()*0.9
    total_max = 0.01
    total_min = -total_max;
    # total_min = abs(z2).min()
    print "  data max = ",total_max
    print "  data min = ",total_min
    print ""

    #total_max = 4500
    #total_min = 1000

    # setup figure (with 3 subplots)
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(10,4))

    zinterval = (total_max - total_min)/5.0

    xmajorLocator   = MultipleLocator(2000)
    ymajorLocator   = MultipleLocator(1000)
    zmajorLocator   = MultipleLocator(zinterval)
    axes.xaxis.set_major_locator(xmajorLocator)
    axes.yaxis.set_major_locator(ymajorLocator)  
    ax = axes

    X, Y, Z = grid(x,y,z2)
    axes.set_ylabel(kernel2)

    #divider = make_axes_locatable(axes)
    axes = plt.gca()
    axes.invert_yaxis()
    #cax = divider.append_axes('right', size='2%', pad=0.05)

    colormap = 'seismic'
        #colormap = 'RdBu'
    #colormap = 'coolwarm'
    #plt.gca().invert_yaxis()
    #plt.style.use('ggplot') 
    my_colormap = make_colormap({0.0:[0.1,0.0,0.0],
                                 0.2:[0.8,0.0,0.0],
                                 0.3:[1.0,0.7,0.0],
                                 0.42:[0.92,0.92,0.92],
                                 0.5:[0.92,0.92,0.92],
                                 0.58:[0.92,0.92,0.92],
                                 0.7:[0.0,0.6,0.7],
                                 0.8:[0.0,0.0,0.8],
                                 1.0:[0.0,0.0,0.1]})

    im = ax.imshow(Z,vmax=total_max, vmin=total_min,
                       extent=[x.min(), x.max(), y.min(), y.max()],cmap=my_colormap)

    # moves plots together
    #fig.subplots_adjust(hspace=0)
    #plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    cbar = fig.colorbar(im,orientation='vertical',fraction=0.016, pad=0.02)

    cbar.set_ticks(zmajorLocator)
    # colorbar
    #fig.colorbar(im, ax=axes.ravel().tolist())
    #fig.colorbar(im, ax=axes.ravel().tolist(),orientation='horizontal')

    # show the figure


    tmp = 0
    snum = "{:03d}".format(tmp)
    fname = 'EigenVector ' + str(snum)
    plt.gca().invert_yaxis()
    plt.title(fname)
    if show:
        plt.figure()

    # saves kernel figure as file
    dir = os.path.dirname(file)
    name_without_ending = str.split(name,".")[0]
    outfile = dir + "./"+ "VP.png"
    fig.savefig(outfile, format="png")

    print "*****"
    print "plotted file: ",outfile
    print "*****"
    print ""


def usage():
    print "usage: ./plot_kernel.py file [1 == show figure / 0 == just plot file]"
    print "   where"
    print "       file - ASCII kernel file, e.g. OUTPUT_FILES/proc000000_rhop_alpha_beta_kernel.dat"

if __name__ == '__main__':
    # gets arguments
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)
    else:
        file = sys.argv[1]

    if len(sys.argv) > 2:
        show_plot = sys.argv[2]
    else:
        show_plot = 0

    if show_plot == '1':
        plot_kernels(file,show=True)
    else:
        plot_kernels(file)

