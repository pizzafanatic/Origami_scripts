#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:40:33 2018

@author: Peter 
"""

# In this script I plot a symbolic representation of a
# 16 * 16 origami pattern consisting out of quadrilaterals
# as well as a real space version of the same pattern


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import tileslibrary as tl


plt.rc('text', usetex=False)
rc('font', **{'size': 16})


def plot_tiling_real_space(coords, raw_tiling, name):
    # Plots the real space version of a tiling
    plt.figure()

    m = coords.shape[0]
    n = coords.shape[1]

    for i in range(m - 1):
        for j in range(n - 2):
            plt.plot(np.array([coords[i, j + 1, 0], coords[i + 1, j + 1, 0]]),
                     np.array([coords[i, j + 1, 1], coords[i + 1, j + 1, 1]]),
                     c='k', lw=0.5, zorder=0)

    for j in range(n - 1):
        for i in range(m - 2):
            plt.plot(np.array([coords[i + 1, j, 0], coords[i + 1, j + 1, 0]]),
                     np.array([coords[i + 1, j, 1], coords[i + 1, j + 1, 1]]),
                     c='k', lw=0.5, zorder=0)

    plt.axis('equal')
    plt.axis('off')

    k = m * n

    for i in range(m - 2):
        for j in range(n - 2):
            if (4 * i == raw_tiling.shape[0]):
                vertex_x = 4 * i - 1
            else:
                vertex_x = 4 * i
            if (4 * j == raw_tiling.shape[1]):
                vertex_y = 4 * j - 1
            else:
                vertex_y = 4 * j
            VERTEX = raw_tiling[vertex_x, vertex_y]
            plt.scatter(coords[i + 1, j + 1, 0],
                        coords[i + 1, j + 1, 1],
                        c='k', s=2500.0 / k, zorder=1)
            if (VERTEX < 5):
                plt.scatter(coords[i + 1, j + 1, 0],
                            coords[i + 1, j + 1, 1],
                            c='w', s=750.0 / k, zorder=2)
            if (VERTEX > 4):
                plt.scatter(coords[i + 1, j + 1, 0], coords[i + 1, j + 1, 1],
                            c='gray', s=750.0 / k, zorder=2)

    plt.savefig(name + '.pdf', format="pdf", transparent=True,
                bbox_inches=0.0, pad_inches=0.0)

    plt.close()


def fill_edge(left_column, top_row):
    # fills two arrays representing the top and left boundary
    # once these are known, the rest of the pattern can be calculated
    # using fill_bulk
    X = np.zeros([4 * len(left_column), 4 * len(top_row)], dtype='int')
    Y = np.empty([len(left_column), len(top_row)], dtype='object')

    if (top_row[0] != left_column[0]):
        print 'HELP'

    code = left_column[0]

    X[0:4, 0:4] = tl.rot_vector(bricks[code[0]][code[1]][code[2]], code[3])
    Y[0, 0] = code

    for i in range(1, len(left_column)):
        code = left_column[i]
        X[4 * i:4 * i + 4, 0:4] = \
            tl.rot_vector(bricks[code[0]][code[1]][code[2]], code[3])
        Y[i, 0] = code
    for i in range(1, len(top_row)):
        code = top_row[i]
        X[0:4, 4 * i:4 * i + 4] = \
            tl.rot_vector(bricks[code[0]][code[1]][code[2]], code[3])
        Y[0, i] = code
    return X, Y


def fill_bulk(X, Y):
    # calculates the remaining (m - 1) * (n - 1) tiles in interior 
    #of the pattern
    height = (X.shape[0] - 4) / 4
    width = (X.shape[1] - 4) / 4
    for i in range(height):
        for j in range(width):
            Toggle = False
            for k in range(140):
                for l in range(4):
                    if (np.array_equal(X[4 + 4 * i:8 + 4 * i,
                                       2 + 4 * j:4 + 4 * j],
                                       tl.monster[k, l, :, :2])):
                        if (np.array_equal(X[2 + 4 * i: 4 + 4 * i,
                                           4 + 4 * j: 8 + 4 * j],
                                           tl.monster[k, l, :2, :])):
                            X[4 + 4 * i: 8 + 4 * i, 4 + 4 * j:8 + 4 * j] = \
                                tl.monster[k, l, :, :]
                            Y[i + 1, j + 1] = tl.convert_dc[k] + \
                                tl.convert_directions[l]
                            Toggle = True
            if (Toggle is False):
                print 'Theres a problem here'
    return X, Y


if __name__ == "__main__":
    bricks = tl.bricks
    tiles_34 = tl.tiles_34

    #Here i specify a 16 by 16 tile pattern, 
    #by specifying the tiles in the top row, and left column

    tp_row = \
        ['f26w', 'f46w', 'f26w', 'c12n', 'c22w', 'c12n', 'f26e', 'f46e',
         'f26e', 'c42e', 'f26w', 'c12n', 'c22w', 'f46w', 'c32n', 'c42w']

    lft_column = \
        ['f26w', 'f26e', 'f26w', 'f26e', 'b10s', 'f46w', 'f46e', 'f46w',
         'f46e', 'b10n', 'f26w', 'f26e', 'b10s', 'f46w', 'f46e', 'b10n']


    tiling_angles, tiling_symbolic = fill_edge(lft_column, tp_row)
    # fill the top and left edge of the symbolic pattern

    tiling_angles, tiling_symbolic = fill_bulk(tiling_angles, tiling_symbolic)
    # finish the pattern by filling in the interior

    name = 'Alpha_Omega_Origami'

    top_lengths = np.array([
        3.774, 9.388, 3.401, 5.024, 2.675, 4.738, 8.657, 3.644, 9.018, 5.056,
        3.176, 5.105, 3.449, 8.621, 5.055, 4.387])

    left_lengths = np.array([
        7.116, 4.144, 8.747, 4.143, 3.085, 3.988, 3.234, 6.38, 3.234, 7.493,
        6.782, 3.52, 3.327, 7.177, 3.833, 7.087])

    TILING = [tiling_angles, tiling_symbolic, name, top_lengths, left_lengths]

    print tiling_symbolic

    prototile = False 
    # whether or not we want to plot the 34 prototiles 
    # or the full set of 140 tiles
    FONTSIZE = 25.0
    #size of the font in the symbolic representation
    CORNERSIZE = 0.25  
    # Size of edge of notch to edge (circle)
    MIDDLESIZE = 0.1 / 2.0  
    # distance between notches
    INDENTSIZE = 0.5 - CORNERSIZE - MIDDLESIZE  
    # depth of notches
    CIRCLE_SIZE = 1 * CORNERSIZE  
    # size of circle indicating angles
    SUP_SIZE = 0.6 * CORNERSIZE  
    # size of circle indicating supplementation
    LINEWIDTH = 0.5
    #linewidth of the mesh


    # Some paramaters for plotting:
    figwidth = tiling_symbolic.shape[1]
    figheight = tiling_symbolic.shape[0]
    figbuff = 0.4
    #figbuff: the white space around the border 

    full_render = True
    # whether or not we want to fully render all the notches and tiles

    FIG = plt.figure(figsize=(figwidth + figbuff, figheight + figbuff),
                     facecolor='none', edgecolor='none', frameon=False)
    AX = FIG.add_axes([0, 0, 1, 1], aspect='equal')
    AX.set_xticklabels([])
    AX.set_yticklabels([])
    plt.axis('off')
    AX.grid(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    for item in [FIG, AX]:
        item.patch.set_visible(False)
    # these settings are necessary to ensure a small boundary 


    # DRAW THE TILING (symbolic)
    for i in range((tiling_symbolic.shape[0])):
        for j in range((tiling_symbolic.shape[1])):
            tile = tiling_symbolic[i, j]
            AX = tl.draw_tile_symbolic(
                AX, bricks, 
                tile[0], tile[1], tile[2], tile[3], 
                j, -i, True, FONTSIZE,
                LINEWIDTH, CORNERSIZE, MIDDLESIZE, SUP_SIZE,
                CIRCLE_SIZE, INDENTSIZE,
                prototile=prototile, full=full_render)

    # SET THE LIMITS OF THE AXIS TO BE JUST RIGHT
    AX.set_xlim([-figbuff / 2., figwidth + figbuff / 2.])
    AX.set_ylim([-figbuff / 2. - figheight, figbuff / 2.])
    AX.set_autoscale_on(False)

    plt.gca().set_aspect(1, adjustable='box')

    #save figure as pdf
    if (full_render is False):
        plt.savefig('big_tiling_symbolisch_draft_mode.pdf', format="pdf",
                    transparent=True, bbox_inches=0.0, pad_inches=0.0)

    elif (full_render is True):
        plt.savefig('big_tiling_symbolisch_fulL_render.pdf', format="pdf",
                    transparent=True, bbox_inches=0.0, pad_inches=0.0)
    plt.close()


    #Now choose angles to convert the symbolic tiling to a real-space pattern
    angles_deg = np.array([60.0, 105.0, 120.0, 75.0])
    # the angles of the vertices that I choose to convert the 
    # symbolic tiling into a real-space tiling
    angles_rad = angles_deg * np.pi / 180.0
    # convert angles to radians

    coords = tl.convert_symbolic_real_space(TILING, angles_rad, top_lengths, left_lengths, 0.5)
    # Her we calculate the coordinates of the vertices 

    plot_tiling_real_space(coords, tiling_angles, name)
    # plot and save a real-space version of the origami pattern 