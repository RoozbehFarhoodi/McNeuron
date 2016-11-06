"""
Basic visualization of neurite morphologies using matplotlib.

Usage is restricted to morphologies in the sWC format with the three-point soma `standard <http://neuromorpho.org/neuroMorpho/SomaFormat.html>`_

"""
import sys,time
import os, sys
from matplotlib.cm import get_cmap
from Crypto.Protocol.AllOrNothing import isInt
sys.setrecursionlimit(10000)
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
import pylab as pl
from matplotlib import collections  as mc
from PIL import Image

from McNeuron import Neuron
from McNeuron import Node

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from numpy import mean,cov,double,cumsum,dot,linalg,array,rank
from pylab import plot,subplot,axis,stem,show,figure, Normalize
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pylab as pl
import matplotlib
from matplotlib import collections  as mc
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

def get_2d_image(path, size, dpi, background, show_width):

    neu = McNeuron.Neuron(file_format = 'swc without attributes', input_file=path)
    depth = neu.location[2,:]
    p = neu.location[0:2,:]
    widths= 5*neu.diameter
    widths[0:3] = 0
    m = min(depth)
    M = max(depth)
    depth =  background * ((depth - m)/(M-m))
    colors = []
    lines = []
    patches = []

    for i in range(neu.n_soma):
        x1 = neu.location[0,i]
        y1 = neu.location[1,i]
        r = 1*neu.diameter[i]
        circle = Circle((x1, y1), r, color = str(depth[i]), ec = 'none',fc = 'none')
        patches.append(circle)

    pa = PatchCollection(patches, cmap=matplotlib.cm.gray)
    pa.set_array(depth[0]*np.zeros(neu.n_soma))

    for i in range(len(neu.nodes_list)):
        colors.append(str(depth[i]))
        j = neu.parent_index[i]
        lines.append([(p[0,i],p[1,i]),(p[0,j],p[1,j])])
    if(show_width):
        lc = mc.LineCollection(lines, colors=colors, linewidths = widths)
    else:
        lc = mc.LineCollection(lines, colors=colors)

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.add_collection(pa)
    fig.set_size_inches([size + 1, size + 1])
    fig.set_dpi(dpi)
    plt.axis('off')
    plt.xlim((min(p[0,:]),max(p[0,:])))
    plt.ylim((min(p[1,:]),max(p[1,:])))
    plt.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    border = (dpi/2)
    return np.squeeze(data[border:-border,border:-border,0])

def projection_on_plane(neuron, normal_vec = np.array([0,0,1]), distance = 10, resolution = np.array([256,256]), gap = 3.0):
    """
    Parameters
    ----------

    return
    ------

    dependency
    ----------
    This function needs following data from neuron:
        location
        diameter
        parent_index
    """
    # projection all the nodes on the plane and finding the right pixel for their centers
    image = np.zeros(resolution)
    shift = resolution[0]/2
    normal_vec1 = np.array([0,0,1])
    normal_vec2 = np.array([0,1,0])
    P = project_points(neuron.location, normal_vec1, normal_vec2)

    for n in neuron.nodes_list:
        if(n.parent != None):
            n1, n2, dis = project_point(n, normal_vec1, normal_vec2)
            pix1 = np.floor(n1/gap) + shift
            pix2 = np.floor(n2/gap) + shift
            if(0 <= pix1 and 0 <= pix2 and pix1<resolution[0] and pix2 < resolution[1]):
                image[pix1, pix2] = dis
    return image

def project_points(location, normal_vectors):
    """
    Parameters
    ----------
    normal_vectors : array of shape [2,3]
        Each row should a normal vector and both of them should be orthogonal.

    location : array of shape [3, n_nodes]
        the location of n_nodes number of points

    Returns
    -------
    cordinates: array of shape [2, n_nodes]
        The cordinates of the location on the plane defined by the normal vectors.
    """
    cordinates = np.dot(normal_vectors, location)
    return cordinates

def depth_points(location, orthogonal_vector):
    """
    Parameters
    ----------
    orthogonal_vector : array of shape [3]
        orthogonal_vector that define the plane

    location : array of shape [3, n_nodes]
        the location of n_nodes number of points

    Returns
    -------
    depth: array of shape [n_nodes]
        The depth of the cordinates when they project on the plane.
    """
    depth = np.dot(orthogonal_vector, location)
    return depth

def make_image(neuron, A, scale_depth, index_neuron):
    normal_vectors = A[0:2,:]
    orthogonal_vector = A[2,:]
    depth = depth_points(neuron.location, orthogonal_vector)
    p = project_points(neuron.location, normal_vectors)
    m = min(depth)
    M = max(depth)
    depth =  scale_depth * ((depth - m)/(M-m))
    colors = []
    lines = []
    for i in range(len(neuron.nodes_list)):
        colors.append((depth[i],depth[i],depth[i],1))
        j = neuron.parent_index[i]
        lines.append([(p[0,i],p[1,i]),(p[0,j],p[1,j])])
    lc = mc.LineCollection(lines, colors=colors, linewidths=2)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    pl.axis('off')
    pl.xlim((min(p[0,:]),max(p[0,:])))
    pl.ylim((min(p[1,:]),max(p[1,:])))
    Name = "neuron" + str(index_neuron[0]+1) + "resample" + str(index_neuron[1]+1) + "angle" + str(index_neuron[2]+1) + ".png"
    fig.savefig(Name,figsize=(6, 6), dpi=80)
    img = Image.open(Name)
    img.load()
    data = np.asarray( img, dtype="int32" )
    data = data[:,:,0]
    return data

def random_unitary_basis(kappa):
    Ax1 = random_2d_rotation_in_3d('x', kappa)
    Ay1 = random_2d_rotation_in_3d('y', kappa)
    Az1 = random_2d_rotation_in_3d('z', kappa)
    Ax2 = random_2d_rotation_in_3d('x', kappa)
    Ay1 = random_2d_rotation_in_3d('y', kappa)
    Az1 = random_2d_rotation_in_3d('z', kappa)
    A = np.dot(np.dot(Ax1,Ay1),Az1)
    B = np.dot(np.dot(Az1,Ay1),Ax1)
    return np.dot(A,B)

def random_2d_rotation_in_3d(axis, kappa):
    theta = np.random.vonmises(0, kappa, 1)
    A = np.eye(3)
    if axis is 'z':
        A[0,0] = np.cos(theta)
        A[1,0] = np.sin(theta)
        A[0,1] = - np.sin(theta)
        A[1,1] = np.cos(theta)
        return A
    if axis is 'y':
        A[0,0] = np.cos(theta)
        A[2,0] = np.sin(theta)
        A[0,2] = - np.sin(theta)
        A[2,2] = np.cos(theta)
        return A
    if axis is 'x':
        A[1,1] = np.cos(theta)
        A[2,1] = np.sin(theta)
        A[1,2] = - np.sin(theta)
        A[2,2] = np.cos(theta)
        return A

def make_six_matrix(A):
    six = []
    six.append(A[[0,1,2],:])
    six.append(A[[0,2,1],:])
    six.append(A[[1,2,0],:])
    six.append(A[[1,0,2],:])
    six.append(A[[2,0,1],:])
    six.append(A[[2,1,0],:])
    return six

def make_six_images(neuron,scale_depth,neuron_index, kappa):
    #A = random_unitary_basis(kappa)
    A = np.eye(3)
    six = make_six_matrix(A)
    D = []
    for i in range(6):
        a = np.append(neuron_index,i)
        D.append(make_image(neuron, six[i], scale_depth, a))
    return D

def generate_data(path, scale_depth, n_camrea, kappa):
    """
    input
    -----
    path : list
        list of all the pathes of swc. each element of the list should be a string.

    scale_depth : float in the interval [0,1]
        a value to differentiate between the background and gray level in the image.

    n_camera : int
        number of different angles to set the six images. For each angle, six images will be generated (up,down and four sides)

    kappa : float
        The width of the distribution that the angles come from. Large value for kappa results in the angles close to x aixs
        kappa = 1 is equvalent to the random angle.

    output
    ------
    Data : list of length
    """
    Data = []
    for i in range(len(path)):
        print path[i]
        neuron = Neuron(file_format = 'swc without attributes', input_file=path[i])
        if(len(neuron.nodes_list) != 0):
            for j in range(n_camrea):
                D = np.asarray(make_six_images(neuron,scale_depth,np.array([i,j]), kappa))
                Data.append(D)
    return Data

def get_all_path(directory):
    fileSet = []
    for root, dirs, files in os.walk(directory):
        for fileName in files:
            if(fileName[-3:] == 'swc'):
                fileSet.append(directory + root.replace(directory, "") + os.sep + fileName)
    return fileSet

def plot_2d(neuron, show_depth, line_width):
    depth = neuron.location[0,:]
    m = min(depth)
    M = max(depth)
    depth = ((depth - m)/(M-m))
    p = neuron.location[0:2,:]
    colors = []
    lines = []
    for i in range(len(neuron.nodes_list)):
        colors.append((depth[i],depth[i],depth[i],1))
        j = neuron.parent_index[i]
        lines.append([(p[0,i],p[1,i]),(p[0,j],p[1,j])])
    if(show_depth == False):
        lc = mc.LineCollection(lines, colors='k', linewidths=line_width)
    else:
        lc = mc.LineCollection(lines, colors=colors, linewidths=line_width)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    pl.axis('off')
    pl.xlim((min(p[0,:]),max(p[0,:])))
    pl.ylim((min(p[1,:]),max(p[1,:])))

def plot_2D(neuron,
            background = 1,
            show_width = False,
            show_depth = False,
            size = 5,
            dpi = 80,
            line_width = 1,
            show_soma = False,
            give_image = False,
            red_after = False,
            node_red = 0,
            translation = (0,0),
            scale_on = False,
            scale = (1,1),
            save = []):

    depth = neuron.location[2,:]
    p = neuron.location[0:2,:]
    if scale_on:
        p[0,:] = scale[0] * (p[0,:]-min(p[0,:]))/(max(p[0,:]) - min(p[0,:]) )
        p[1,:] = scale[1] * (p[1,:]-min(p[1,:]))/(max(p[1,:]) - min(p[1,:]) )
    widths= neuron.diameter
    #widths[0:3] = 0
    m = min(depth)
    M = max(depth)
    depth =  background * ((depth - m)/(M-m))
    colors = []
    lines = []
    patches = []

    for i in range(neuron.n_soma):
        x1 = neuron.location[0,i] + translation[0]
        y1 = neuron.location[1,i] + translation[1]
        r = widths[i]
        circle = Circle((x1, y1), r, color = str(depth[i]), ec = 'none',fc = 'none')
        patches.append(circle)

    pa = PatchCollection(patches, cmap=matplotlib.cm.gray)
    pa.set_array(depth[0]*np.zeros(neuron.n_soma))

    for i in range(len(neuron.nodes_list)):
        colors.append(str(depth[i]))
        j = neuron.parent_index[i]
        lines.append([(p[0,i] + translation[0],p[1,i] + translation[1]),(p[0,j] + translation[0],p[1,j] + translation[1])])

    if(show_width):
        if(show_depth):
            lc = mc.LineCollection(lines, colors=colors, linewidths = line_width*widths)
        else:
            lc = mc.LineCollection(lines, linewidths = line_width*widths)
    else:
        if(show_depth):
            lc = mc.LineCollection(lines, colors=colors, linewidths = line_width)
        else:
            lc = mc.LineCollection(lines, linewidths = line_width, color = 'k')

    if(give_image):
        if(red_after):
            line1 = []
            line2 = []
            (I1,) = np.where(~np.isnan(neuron.connection[:,node_red]))
            (I2,) = np.where(np.isnan(neuron.connection[:,node_red]))
            for i in I1:
                j = neuron.parent_index[i]
                line1.append([(p[0,i],p[1,i]),(p[0,j],p[1,j])])
                lc1 = mc.LineCollection(line1, linewidths = 2*line_width, color = 'r')
            for i in I2:
                j = neuron.parent_index[i]
                line2.append([(p[0,i],p[1,i]),(p[0,j],p[1,j])])
                lc2 = mc.LineCollection(line2, linewidths = line_width, color = 'k')
            return (lc1, lc2, (min(p[0,:]),max(p[0,:])), (min(p[1,:]),max(p[1,:])))
        else:
            return (lc, (min(p[0,:]),max(p[0,:])), (min(p[1,:]),max(p[1,:])))
    else:
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        if(show_soma):
            ax.add_collection(pa)
        fig.set_size_inches([size + 1, size + 1])
        fig.set_dpi(dpi)
        plt.axis('off')
        plt.xlim((min(p[0,:]),max(p[0,:])))
        plt.ylim((min(p[1,:]),max(p[1,:])))
        plt.draw()
        if(len(save)!=0):
            plt.savefig(save, format = "eps")


# def plot_2D(neuron, background = 1, show_width = False, show_depth = False , size = 5, dpi = 80, line_width = 1):
#     depth = neuron.location[2,:]
#     p = neuron.location[0:2,:]
#     widths= neuron.diameter
#     m = min(depth)
#     M = max(depth)
#     depth =  background * ((depth - m)/(M-m))
#     colors = []
#     lines = []
#     patches = []
#
#     for i in range(neuron.n_soma):
#         x1 = neuron.location[0,i]
#         y1 = neuron.location[1,i]
#         r = neuron.diameter[i]
#         circle = Circle((x1, y1), r, color = str(depth[i]), ec = 'none',fc = 'none')
#         patches.append(circle)
#
#     pa = PatchCollection(patches, cmap=matplotlib.cm.gray)
#     pa.set_array(depth[0]*np.zeros(neuron.n_soma))
#
#     for i in range(len(neuron.nodes_list)):
#         colors.append(str(depth[i]))
#         j = neuron.parent_index[i]
#         lines.append([(p[0,i],p[1,i]),(p[0,j],p[1,j])])
#     if(show_width):
#         if(show_depth):
#             lc = mc.LineCollection(lines, colors=colors, linewidths = line_width*widths)
#         else:
#             lc = mc.LineCollection(lines, linewidths = line_width*widths)
#     else:
#         if(show_depth):
#             lc = mc.LineCollection(lines, colors=colors, linewidths = line_width)
#         else:
#             lc = mc.LineCollection(lines, linewidths = line_width)
#
#     fig, ax = plt.subplots()
#     ax.add_collection(lc)
#     #ax.add_collection(pa)
#     fig.set_size_inches([size + 1, size + 1])
#     fig.set_dpi(dpi)
#     plt.axis('off')
#     plt.xlim((min(p[0,:]),max(p[0,:])))
#     plt.ylim((min(p[1,:]),max(p[1,:])))
#     plt.draw()
#     return fig

def plot_3D(neuron, color_scheme="default", color_mapping=None,
            synapses=None, save_image="animation",show_radius=True):
    """
    3D matplotlib plot of a neuronal morphology. The SWC has to be formatted with a "three point soma".
    Colors can be provided and synapse location marked

    Parameters
    -----------
    color_scheme: string
        "default" or "neuromorpho". "neuronmorpho" is high contrast
    color_mapping: list[float] or list[list[float,float,float]]
        Default is None. If present, this is a list[N] of colors
        where N is the number of compartments, which roughly corresponds to the
        number of lines in the SWC file. If in format of list[float], this list
        is normalized and mapped to the jet color map, if in format of
        list[list[float,float,float,float]], the 4 floats represt R,G,B,A
        respectively and must be between 0-255. When not None, this argument
        overrides the color_scheme argument(Note the difference with segments).
    synapses : vector of bools
        Default is None. If present, draw a circle or dot in a distinct color
        at the location of the corresponding compartment. This is a
        1xN vector.
    save_image: string
        Default is None. If present, should be in format "file_name.extension",
        and figure produced will be saved as this filename.
    show_radius : boolean
        True (default) to plot the actual radius. If set to False,
        the radius will be taken from `btmorph2\config.py`
    """

    if show_radius==False:
        plot_radius = config.fake_radius

    if color_scheme == 'default':
        my_color_list = config.c_scheme_default['neurite']
    elif color_scheme == 'neuromorpho':
        my_color_list = config.c_scheme_nm['neurite']
    else:
        raise Exception("Not valid color scheme")
    #print 'my_color_list: ', my_color_list

    fig, ax = plt.subplots()

    if color_mapping is not None:
        if isinstance(color_mapping[0], int):
            jet = plt.get_cmap('jet')
            norm = colors.Normalize(np.min(color_mapping), np.max(color_mapping))
            scalarMap = cm.ScalarMappable(norm=norm, cmap=jet)

            Z = [[0, 0], [0, 0]]
            levels = np.linspace(np.min(color_mapping), np.max(color_mapping), 100)
            CS3 = plt.contourf(Z, levels, cmap=jet)
            plt.clf()

    ax = fig.gca(projection='3d')

    index = 0

    for node in neuron.nodes_list: # not ordered but that has little importance here
        # draw a line segment from parent to current point
        c_x = node.xyz[0]
        c_y = node.xyz[1]
        c_z = node.xyz[2]
        c_r = node.r

        if index < 3:
            pass
        else:
            parent = node.parent
            p_x = parent.xyz[0]
            p_y = parent.xyz[1]
            p_z = parent.xyz[2]
            # p_r = parent.content['p3d'].radius
            # print 'index:', index, ', len(cs)=', len(color_mapping)
            if show_radius==False:
                line_width = plot_radius
            else:
                line_width = c_r/2.0

            if color_mapping is None:
                ax.plot([p_x, c_x], [p_y, c_y], [p_z, c_z], my_color_list[node.set_type_from_name() - 1], linewidth=line_width)
            else:
                if isinstance(color_mapping[0], int):
                    c = scalarMap.to_rgba(color_mapping[index])
                elif isinstance(color_mapping[0], list):
                    c = [float(x) / 255 for x in color_mapping[index]]

                ax.plot([p_x, c_x], [p_y, c_y], [p_z, c_z], c=c, linewidth=c_r/2.0)
            # add the synapses
        if synapses is not None:
            if synapses[index]:
                ax.scatter(c_x, c_y, c_z, c='r')

        index += 1

    #minv, maxv = neuron.get_boundingbox()
    #minv = min(minv)
    #maxv = max(maxv)
    #ax.auto_scale_xyz([minv, maxv], [minv, maxv], [minv, maxv])

    index = 0

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if color_mapping is not None:
        if isinstance(color_mapping[0], int):
            cb = plt.colorbar(CS3) # bit of a workaround, but it seems to work
            ticks_f = np.linspace(np.min(color_mapping)-1, np.max(color_mapping)+1, 5)
            ticks_i = map(int, ticks_f)
            cb.set_ticks(ticks_i)

    # set the bg color
    fig = plt.gcf()
    ax = fig.gca()
    if color_scheme == 'default':
        ax.set_axis_bgcolor(config.c_scheme_default['bg'])
    elif color_scheme == 'neuromorpho':
        ax.set_axis_bgcolor(config.c_scheme_nm['bg'])

    if save_image is not None:
        plt.savefig(save_image)

    plt.show()

    return fig

def animate(neuron, color_scheme="default", color_mapping=None,
            synapses=None, save_image=None, axis="z"):
    """
    3D matplotlib plot of a neuronal morphology. The SWC has to be formatted with a "three point soma".
    Colors can be provided and synapse location marked

    Parameters
    -----------
    color_scheme: string
        "default" or "neuromorpho". "neuronmorpho" is high contrast
    color_mapping: list[float] or list[list[float,float,float]]
        Default is None. If present, this is a list[N] of colors
        where N is the number of compartments, which roughly corresponds to the
        number of lines in the SWC file. If in format of list[float], this list
        is normalized and mapped to the jet color map, if in format of
        list[list[float,float,float,float]], the 4 floats represt R,G,B,A
        respectively and must be between 0-255. When not None, this argument
        overrides the color_scheme argument(Note the difference with segments).
    synapses : vector of bools
        Default is None. If present, draw a circle or dot in a distinct color
        at the location of the corresponding compartment. This is a
        1xN vector.
    save_image: string
        Default is None. If present, should be in format "file_name.extension",
        and figure produced will be saved as this filename.

    """

    if color_scheme == 'default':
        my_color_list = config.c_scheme_default['neurite']
    elif color_scheme == 'neuromorpho':
        my_color_list = config.c_scheme_nm['neurite']
    else:
        raise Exception("Not valid color scheme")
    print 'my_color_list: ', my_color_list

    fig, ax = plt.subplots()

    if color_mapping is not None:
        if isinstance(color_mapping[0], int):
            jet = plt.get_cmap('jet')
            norm = colors.Normalize(np.min(color_mapping), np.max(color_mapping))
            scalarMap = cm.ScalarMappable(norm=norm, cmap=jet)

            Z = [[0, 0], [0, 0]]
            levels = np.linspace(np.min(color_mapping), np.max(color_mapping), 100)
            CS3 = plt.contourf(Z, levels, cmap=jet)
            plt.clf()

    ax = fig.gca(projection='3d')

    index = 0

    for node in neuron.nodes_list: # not ordered but that has little importance here
        # draw a line segment from parent to current point
        c_x = node.xyz[0]
        c_y = node.xyz[1]
        c_z = node.xyz[2]
        c_r = node.r

        if index < 3:
            pass
        else:
            parent = node.parent
            p_x = parent.xyz[0]
            p_y = parent.xyz[1]
            p_z = parent.xyz[2]
            # p_r = parent.content['p3d'].radius
            # print 'index:', index, ', len(cs)=', len(color_mapping)
            if color_mapping is None:
                ax.plot([p_x, c_x], [p_y, c_y], [p_z, c_z], my_color_list[node.set_type_from_name() - 1], linewidth=c_r/2.0)
            else:
                if isinstance(color_mapping[0], int):
                    c = scalarMap.to_rgba(color_mapping[index])
                elif isinstance(color_mapping[0], list):
                    c = [float(x) / 255 for x in color_mapping[index]]

                ax.plot([p_x, c_x], [p_y, c_y], [p_z, c_z], c=c, linewidth=c_r/2.0)
            # add the synapses
        if synapses is not None:
            if synapses[index]:
                ax.scatter(c_x, c_y, c_z, c='r')

        index += 1

    #minv, maxv = neuron.get_boundingbox()
    #minv = min(minv)
    #maxv = max(maxv)
    #ax.auto_scale_xyz([minv, maxv], [minv, maxv], [minv, maxv])

    index = 0

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if color_mapping is not None:
        if isinstance(color_mapping[0], int):
            cb = plt.colorbar(CS3) # bit of a workaround, but it seems to work
            ticks_f = np.linspace(np.min(color_mapping)-1, np.max(color_mapping)+1, 5)
            ticks_i = map(int, ticks_f)
            cb.set_ticks(ticks_i)

    # set the bg color
    fig = plt.gcf()
    ax = fig.gca()
    if color_scheme == 'default':
        ax.set_axis_bgcolor(config.c_scheme_default['bg'])
    elif color_scheme == 'neuromorpho':
        ax.set_axis_bgcolor(config.c_scheme_nm['bg'])

    anim = animation.FuncAnimation(fig, _animate_rotation,fargs=(ax,), frames=60)
    #anim.save(save_image + ".gif", writer='imagemagick', fps=4)

    # anim.save(save_image + ".gif", writer='ffmpeg', fps=4)


    return fig

def _animate_rotation(nframe,fargs):
    fargs.view_init(elev=0, azim=nframe*6)

def plot_3D_Forest(neuron, color_scheme="default", save_image=None):
    """
    3D matplotlib plot of a neuronal morphology. The Forest has to be formatted with a "three point soma".
    Colors can be provided and synapse location marked

    Parameters
    -----------
    color_scheme: string
        "default" or "neuromorpho". "neuronmorpho" is high contrast
    save_image: string
        Default is None. If present, should be in format "file_name.extension",
        and figure produced will be saved as this filename.
    """
    my_color_list = ['r','g','b','c','m','y','r--','b--','g--']

    # resolve some potentially conflicting arguments
    if color_scheme == 'default':
        my_color_list = config.c_scheme_default['neurite']
    elif color_scheme == 'neuromorpho':
        my_color_list = config.c_scheme_nm['neurite']
    else:
        raise Exception("Not valid color scheme")
    print 'my_color_list: ', my_color_list

    fig, ax = plt.subplots()

    ax = fig.gca(projection='3d')


    index = 0
    for node in neuron.nodes_list:
        c_x = node.xyz[0]
        c_y = node.xyz[1]
        c_z = node.xyz[2]
        c_r = node.r

        if index < 3:
            pass
        else:
            parent = node.parent
            p_x = parent.xyz[0]
            p_y = parent.xyz[1]
            p_z = parent.xyz[2]
            # p_r = parent.content['p3d'].radius
            # print 'index:', index, ', len(cs)=', len(color_mapping)

            ax.plot([p_x, c_x], [p_y, c_y], [p_z, c_z], my_color_list[node.set_type_from_name() - 1], linewidth=c_r/2.0)
        index += 1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    if save_image is not None:
        plt.savefig(save_image)

    return fig

def plot_dendrogram(neuron,transform='plain',shift=0,c='k',radius=True,rm=20000.0,ra=200,outN=None) :
    """
    Generate a dendrogram from an SWC file. The SWC has to be formatted with a "three point soma"

    Parameters
    -----------
    file_name : string
        File name of the SWC file to plots
    transform : string
        Either 'plain' or 'lambda'. Plain means no transform while 'lambda' performs an elecotrtonic transform
    shift : float
        Offset in the x-direction
    c : string
        Color ('r','g', 'b', ...)
    radius : boolean
        Plot a wire (False) dendrogram or one with the thickness of the processes (True)
    rm : float
       Membrane resistance. Only needed when transform = 'lambda'
    rm : float
       Axial resistance. Only needed when transform = 'lambda'
    outN : string
        File name of the output file. Extension of this file sets the file type
    """
    global C, RM, RA, max_width, max_height # n.a.s.t.y.

    swc_tree = neuron.tree
    RM = rm
    RA = ra
    C = c
    max_height = 0
    max_width = 0
    plt.clf()
    print 'Going to build the dendrogram. This might take a while...'
    ttt = time.time()
    _expand_dendrogram(swc_tree.root,swc_tree,shift,0,radius=radius,transform=transform)
    if(transform == 'plain') :
        plt.ylabel('L (micron)')
    elif(transform == 'lambda') :
        plt.ylabel('L (lambda)')
    print (time.time() - ttt), ' later the dendrogram was finished. '

    print 'max_widht=%f, max_height=%f' % (max_width,max_height)
    x_bound = (max_width / 2.0) + (0.1*max_width)
    max_y_bound = max_height + 0.1*max_height
    plt.axis([-1.0*x_bound,x_bound,-0.1*max_height,max_y_bound])

    plt.plot([x_bound,x_bound],[0,100],'k', linewidth=5) # 250 for MN, 100 for granule

    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    if(outN != None) :
        plt.savefig(outN)

def _expand_dendrogram(cNode,swc_tree,off_x,off_y,radius,transform='plain') :
    global max_width,max_height # middle name d.i.r.t.y.
    '''
    Gold old fashioned recursion... sys.setrecursionlimit()!
    '''
    place_holder_h = H_SPACE
    max_degree = swc_tree.degree_of_node(cNode)
    required_h_space = max_degree * place_holder_h
    start_x = off_x-(required_h_space/2.0)
    if(required_h_space > max_width) :
        max_width = required_h_space

    if swc_tree.is_root(cNode) :
        print 'i am expanding the root'
        cNode.children.remove(swc_tree.get_node_with_index(2))
        cNode.children.remove(swc_tree.get_node_with_index(3))

    for cChild in cNode.children :
        l = _path_between(swc_tree,cChild,cNode,transform=transform)
        r = cChild.content['p3d'].radius

        cChild_degree = swc_tree.degree_of_node(cChild)
        new_off_x = start_x + ( (cChild_degree/2.0)*place_holder_h )
        new_off_y = off_y+(V_SPACE*2)+l
        r = r if radius  else 1
        plt.vlines(new_off_x,off_y+V_SPACE,new_off_y,linewidth=r,colors=C)
        if((off_y+(V_SPACE*2)+l) > max_height) :
            max_height = off_y+(V_SPACE*2)+l

        _expand_dendrogram(cChild,swc_tree,new_off_x,new_off_y,radius=radius,transform=transform)

        start_x = start_x + (cChild_degree*place_holder_h)
        plt.hlines(off_y+V_SPACE,off_x,new_off_x,colors=C)

def _path_between(swc_tree,deep,high,transform='plain') :
    path = swc_tree.path_to_root(deep)
    pl = 0
    pNode = deep
    for node in path[1:] :
        pPos = pNode.content['p3d'].xyz
        cPos = node.content['p3d'].xyz
        pl = pl + np.sqrt(np.sum((cPos-pPos)**2))
        #pl += np.sqrt( (pPos.x - cPos.x)**2 + (pPos.y - cPos.y)**2 + (pPos.z - cPos.z)**2 )
        pNode = node
        if(node == high) : break

    if(transform == 'plain'):
        return pl
    elif(transform == 'lambda') :
        DIAM = (deep.content['p3d'].radius*2.0 + high.content['p3d'].radius*2.0) /2.0 # naive...
        c_lambda = np.sqrt(1e+4*(DIAM/4.0)*(RM/RA))
        return pl / c_lambda
