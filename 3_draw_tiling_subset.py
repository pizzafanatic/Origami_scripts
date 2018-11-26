#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:40:33 2018

@author: Peter 
"""

# In this script I plo t

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import tileslibrary as tl


plt.rc('text', usetex=False)
rc('font', **{'size': 16})


def plot_tiling_real_space(coords, raw_tiling, name):
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
    X = np.zeros([4 * len(left_column), 4 * len(top_row)], dtype='int')
    Y = np.empty([len(left_column), len(top_row)], dtype='object')

    if (top_row[0] != left_column[0]):
        print 'HELP'

    code = left_column[0]

    print code[0]
    print code[1]
    print code[2]
    print code[3]
    print bricks[code[0]][code[1]][code[2]]

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
                                convert_directions[l]
                            Toggle = True
            if (Toggle is False):
                print 'Theres a problem here'
    return X, Y


if __name__ == "__main__":
    bricks = tl.bricks
    tiles_34 = tl.tiles_34

    convert_directions = {0: 'n', 1: 'w', 2: 's', 3: 'e'}

    tp_row = \
        ['f26w', 'f46w', 'f26w', 'c12n', 'c22w', 'c12n', 'f26e', 'f46e',
         'f26e', 'c42e', 'f26w', 'c12n', 'c22w', 'f46w', 'c32n', 'c42w']

    lft_column = \
        ['f26w', 'f26e', 'f26w', 'f26e', 'b10s', 'f46w', 'f46e', 'f46w',
         'f46e', 'b10n', 'f26w', 'f26e', 'b10s', 'f46w', 'f46e', 'b10n']

    X, Y = fill_edge(lft_column, tp_row)
    X, Y = fill_bulk(X, Y)

    Z = 'Alpha_Omega_Origami'

    top = np.array([
        3.774, 9.388, 3.401, 5.024, 2.675, 4.738, 8.657, 3.644, 9.018, 5.056,
        3.176, 5.105, 3.449, 8.621, 5.055, 4.387])

    left = np.array([
        7.116, 4.144, 8.747, 4.143, 3.085, 3.988, 3.234, 6.38, 3.234, 7.493,
        6.782, 3.52, 3.327, 7.177, 3.833, 7.087])

    packed = [X, Y, Z, top, left]

    print Y

    prototile = False
    FONTSIZE = 25.0
    linewidth_2 = .25

    CORNERSIZE = 0.25  # Size of edge of notch to edge (circle)
    MIDDLESIZE = 0.1 / 2.0  # Ass. w/ size between notches
    INDENTSIZE = 0.5 - CORNERSIZE - MIDDLESIZE  # Ass. w/ size of notche
    CIRCLE_SIZE = 1 * CORNERSIZE  # Clearly circle size
    SUP_SIZE = 0.6 * CORNERSIZE  # Suppl. circle size
    LINEWIDTH = 2 * linewidth_2

    FUDGE_FACTOR_2 = 0.0
    FACTOR = 0.0

    convert_dir = {'n': 0, 'w': 1, 's': 2, 'e': 3}
    tiling = packed[1]
    raw_tiling = packed[0]
    Z = packed[2]

    figwidth = tiling.shape[1]
    figheight = tiling.shape[0]
    figbuff = 0.4

    full_render = True

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

    eerste = tiling[0, 0]

    # DRAW THE TILING
    for i in range((tiling.shape[0])):
        for j in range((tiling.shape[1])):
            code = tiling[i, j]
            AX = tl.draw_tile_symbolic(
                AX, bricks, code[0], code[1], code[2],
                code[3], j, -i, True, FONTSIZE,
                LINEWIDTH, CORNERSIZE, MIDDLESIZE, SUP_SIZE,
                CIRCLE_SIZE, INDENTSIZE,
                prototile=prototile, full=full_render)

    # SET THE LIMITS OF THE AXIS TO BE JUST RIGHT
    AX.set_xlim([-figbuff / 2., figwidth + figbuff / 2.])
    AX.set_ylim([-figbuff / 2. - figheight, figbuff / 2.])
    AX.set_autoscale_on(False)

    plt.gca().set_aspect(1, adjustable='box')

    if (full_render is False):
        plt.savefig('big_tiling_symbolisch_draft_mode.pdf', format="pdf",
                    transparent=True, bbox_inches=0.0, pad_inches=0.0)

    elif (full_render is True):
        plt.savefig('big_tiling_symbolisch_fulL_render.pdf', format="pdf",
                    transparent=True, bbox_inches=0.0, pad_inches=0.0)
    plt.close()

    tiling = packed
    Z = tiling[2]

    angles_deg = np.array([60.0, 105.0, 120.0, 75.0])
    angles_rad = angles_deg * np.pi / 180.0
    top = tiling[3]
    left = tiling[4]
    coords = tl.convert_symbolic_real_space(tiling, angles_rad, top, left, 0.5)
    tl.output_csv_coords(coords, Z, 15)
    tiling = packed[1]
    raw_tiling = packed[0]

    plot_tiling_real_space(coords, raw_tiling, Z)
