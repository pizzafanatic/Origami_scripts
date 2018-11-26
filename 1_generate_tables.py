#!/usr/bin/env python2
#  - * -  coding: utf - 8  - * - 
"""
Created on Tue Jan 16 15:40:33 2018

@author: Peter
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from copy import deepcopy
import tileslibrary as tl


plt.rc('text', usetex=False)
rc('font', **{'size': 16})


def fill_bulk(X, Y):
    height = (X.shape[0] - 4) / 4
    width = (X.shape[1] - 4) / 4

    for i in range(height):
        for j in range(width):
            Toggle = False
            for k in range(140):
                for l in range(4):
                    if(np.array_equal(
                       X[4 + 4 * i: 8 + 4 * i,
                         2 + 4 * j:4 + 4 * j], tl.monster[k, l, :, :2])):
                        if(np.array_equal(X[2 + 4 * i:4 + 4 * i,
                                            4 + 4 * j:8 + 4 * j],
                           tl.monster[k, l, :2, :])):
                            X[4 + 4 * i: 8 + 4 * i, 4 + 4 * j: 8 + 4 * j] =\
                                tl.monster[k, l, :, :]
                            Y[i + 1, j + 1] = tl.convert_dc[k] +\
                                tl.convert_directions[l]
                            Toggle = True
            if(Toggle is False):
                print 'Theres a problem here'
    return X, Y


def draw_tile_realspace(ax,bricks, bricktype, index, sup_index, orientation,
                        xpos, ypos, letter, font_size, tiling, angles_rad,
                        top, left, settings, linewidth, vertexsize, linewidth_vertex):

    convert_directions = {'n': 0, 'w': 1, 's': 2, 'e': 3}
    brick = bricks[bricktype][index][sup_index]
    brick = np.rot90(brick, k=convert_directions[orientation])
    bricktype = bricktype.upper()
    relabel = {'0': '1',
               '7': '2',
               '3': '3',
               '5': '4',
               '1': '5',
               '2': '6',
               '4': '7',
               '6': '8'}

    name = bricktype + '_{' + index + '}' + '^' + relabel[sup_index]

    if(letter is True):
        ax.text(xpos + settings['xoffset_letter'] + settings['global_x_shift'],
                ypos - settings['yoffset_letter'] + settings['global_y_shift'],
                r'$' + name + '$',
                fontsize=FONTSIZE,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=90.0 * convert_directions[orientation])

    coords = tl.draw_tiling(tiling,
                            angles_rad,
                            top,
                            left,
                            settings['dangling_ends'])
    coords = coords / 1.2
    coords[:, :, 0] = coords[:, :, 0] + xpos + settings['global_x_shift']
    coords[:, :, 1] = coords[:, :, 1] + ypos + settings['global_y_shift']
    Z = ' '
    # def plot_tiling(ax, coords, raw_tiling, name, linewidth, 
    #             vertexsize, linewidth_vertex):
    ax = tl.plot_tiling(ax, coords, tiling[0], Z, linewidth, vertexsize, linewidth_vertex)
    return ax


def fill_edge(left_column, top_row):
    X = np.zeros([4 * len(left_column), 4 * len(top_row)], dtype='int')
    Y = np.empty([len(left_column), len(top_row)], dtype='object')
    if(top_row[0] != left_column[0]):
        print 'HELP'

    code = left_column[0]

    X[0:4, 0:4] = tl.rot_vector(bricks[code[0]][code[1]][code[2]],
                                code[3])
    Y[0, 0] = code

    for i in range(1, len(left_column)):
        code = left_column[i]
        X[4 * i:4 * i + 4, 0:4] =\
            tl.rot_vector(bricks[code[0]][code[1]][code[2]],
                          code[3])
        Y[i, 0] = code

    for i in range(1, len(top_row)):
        code = top_row[i]

        X[0:4, 4 * i: 4 * i + 4] =\
            tl.rot_vector(bricks[code[0]][code[1]][code[2]],
                          code[3])
        Y[0, i] = code
    return X, Y


def plot_mesh(ax, type_tegel, normalsups, indices, table_pos, settings, 
              bricklib, angles_grad, tablespacing, fontsize,
              linewidth, vertexsize, linewidth_vertex):

    xmultiplier = table_pos[0]
    xoffset = table_pos[1]
    ymultiplier = table_pos[2]
    yoffset = table_pos[3]
    row_sum_1 = table_pos[4]
    row_sum_2 = table_pos[5]

    for letter in range(len(type_tegel)):
        row = 0
        for sup in normalsups:
            for index in indices:
                tp_row = [type_tegel[letter] + index + sup + 'n']
                lft_column = [type_tegel[letter] + index + sup + 'n']
                X, Y = fill_edge(lft_column, tp_row)
                X, Y = fill_bulk(X, Y)
                top = np.array([settings[type_tegel[letter]]
                               [index][sup]['length_top']])
                left = np.array([settings[type_tegel[letter]]
                                [index][sup]['length_left']])
                tiling = [X, Y, top, left]
                angles_rad = angles_grad * np.pi / 180.0
                top = tiling[2]
                left = tiling[3]
                ax = draw_tile_realspace(ax,
                    bricklib, type_tegel[letter], index, sup, 'n',
                    tablespacing * (xmultiplier * letter + xoffset),
                    -tablespacing * 1.2 *
                    (ymultiplier * row + yoffset),
                    True, fontsize, tiling, angles_rad, top, left,
                    settings[type_tegel[letter]][index][sup],
                    linewidth, vertexsize, linewidth_vertex)
                row = row + row_sum_1
            row = row + row_sum_2
    return ax


if __name__ == "__main__":
    # D
    # 
    bricks = tl.bricks
    tiles_34 = tl.tiles_34

    m = 10
    n = 16

    bumper = 0.4

    tablespacing = 1.4
    factor_canvas = 0.25

    figwidth = float(m)
    figheight = float(n)


    LINEWIDTH_2 = 0.8 / (m * m) ** 0.25
    MIDDLESIZE = 0.1 / 2.0
    CORNERSIZE = 0.25
    INDENTSIZE = 0.5 - CORNERSIZE - MIDDLESIZE
    FACTOR = 0.05
    FACTOR = 0.00
    CIRCLE_SIZE = 1 * CORNERSIZE
    LINEWIDTH = 3 * LINEWIDTH_2
    FUDGE_FACTOR_2 = 0.0
    INDENTSIZE = 0.5 - CORNERSIZE - MIDDLESIZE
    SUP_SIZE = 0.6 * CORNERSIZE
    fontsize = 20


    # LINEWIDTH, CORNERSIZE, MIDDLESIZE, SUP_SIZE, CIRCLE_SIZE, INDENTSIZE, FACTOR, FUDGE_FACTOR_2)


    print 'inches: ' + str(figwidth) + ' mm: ' + str(figwidth * 25.4)
    print 'inches: ' + str(figheight) + ' mm: ' + str(figheight * 25.4)

    FIG1 = plt.figure(figsize=(
                     figwidth + bumper,
                     figheight + bumper),
                     facecolor='none',
                     edgecolor='none', frameon=False)

    AX1 = FIG1.add_axes([0, 0, 1, 1], aspect='equal')

    type_tegel = ['d', 'e', 'f', 'g', 'h']
    normalsups = ['1', '2', '4', '6']
    indices = ['1', '2', '3', '4']

    for i in range(len(type_tegel)):
        row = 0
        for j in normalsups:
            for k in indices:
                AX1 = tl.draw_tile_symbolic(AX1, bricks, type_tegel[i], k, j, 'n',
                                   tablespacing * (i + 2),
                                   -tablespacing * float(row), True, fontsize,
                                   LINEWIDTH, CORNERSIZE, MIDDLESIZE, 
                                   SUP_SIZE, CIRCLE_SIZE, INDENTSIZE, 
                                   FACTOR, FUDGE_FACTOR_2, prototile=False, full=True)
                row = row + 1

    type_tegel = ['i', 'k']
    normalsups = ['1', '2', '4', '6']
    indices = ['1', '2']

    for i in range(len(type_tegel)):
        row = 0
        for j in normalsups:
            for k in indices:
                AX1 = tl.draw_tile_symbolic(AX1, bricks, type_tegel[i], k, j, 'n',
                                   tablespacing * (2 * i + 7),
                                   -tablespacing * float(row),
                                   True, fontsize,
                                   LINEWIDTH, CORNERSIZE, MIDDLESIZE, 
                                   SUP_SIZE, CIRCLE_SIZE, INDENTSIZE, 
                                   FACTOR, FUDGE_FACTOR_2, prototile=False, full=True)
                row = row + 1
            row = row + 2

    type_tegel = ['j']

    normalsups = ['1', '2', '4', '6']
    indices = ['1', '2', '3', '4']


    for i in range(len(type_tegel)):
        row = 0
        for j in normalsups:
            for k in indices:
                AX1 = tl.draw_tile_symbolic(AX1,bricks, type_tegel[i], k, j, 'n',
                                   tablespacing * (i + 8), 
                                   - tablespacing * float(row),
                                   True, fontsize,
                                   LINEWIDTH, CORNERSIZE, MIDDLESIZE, 
                                   SUP_SIZE, CIRCLE_SIZE, INDENTSIZE, 
                                   FACTOR, FUDGE_FACTOR_2, prototile=False, full=True)
                row = row + 1

    type_tegel = ['c']
    normalsups = ['1', '2', '4']
    indices = ['1', '2', '3', '4']

    for i in range(len(type_tegel)):
        row = 0
        for j in normalsups:
            for k in indices:
                AX1 = tl.draw_tile_symbolic(AX1,bricks, type_tegel[i], k, j, 'n',
                                   tablespacing * (i + 1),
                                   - tablespacing * float(row),
                                   True, fontsize,
                                   LINEWIDTH, CORNERSIZE, MIDDLESIZE, 
                                   SUP_SIZE, CIRCLE_SIZE, INDENTSIZE, 
                                   FACTOR, FUDGE_FACTOR_2, prototile=False, full=True)                
                row = row + 1
    type_tegel = ['a']
    normalsups = ['0', '7']
    indices = ['1']

    for i in range(len(type_tegel)):
        row = 0
        for j in normalsups:
            for k in indices:
                AX1 = tl.draw_tile_symbolic(AX1, bricks, type_tegel[i], k, j, 'n',
                          tablespacing * (i), 
                          -tablespacing * 2 * float(row), True, fontsize, 
                           LINEWIDTH, CORNERSIZE, MIDDLESIZE, 
                           SUP_SIZE, CIRCLE_SIZE, INDENTSIZE, 
                           FACTOR, FUDGE_FACTOR_2, prototile=False, full=True)                
                row = row + 1

    type_tegel = ['b']
    normalsups = ['0', '7']
    indices = ['1']

    for i in range(len(type_tegel)):
        row = 0
        for j in normalsups:
            for k in indices:
                AX1 = tl.draw_tile_symbolic(AX1, bricks, type_tegel[i], k, j, 'n', 
                                   tablespacing * (i),
                                   - tablespacing * (2 * row + 1),
                                   True, fontsize,
                                   LINEWIDTH, CORNERSIZE, MIDDLESIZE, 
                                   SUP_SIZE, CIRCLE_SIZE, INDENTSIZE, 
                                   FACTOR, FUDGE_FACTOR_2, prototile=False, full=True)                
                row = row + 1

    type_tegel = ['c']
    normalsups = ['3']
    indices = ['1', '2', '3', '4']

    for i in range(len(type_tegel)):
        row = 0
        for j in normalsups:
            for k in indices:
                AX1 = tl.draw_tile_symbolic(AX1, bricks, type_tegel[i], k, j, 'n',
                                   tablespacing * (i),
                                   - tablespacing * float(row + 4),
                                   True, fontsize,
                                   LINEWIDTH, CORNERSIZE, MIDDLESIZE, 
                                   SUP_SIZE, CIRCLE_SIZE, INDENTSIZE, 
                                   FACTOR, FUDGE_FACTOR_2, prototile=False, full=True)
                row = row + 1

    type_tegel = ['d']
    normalsups = ['3']
    indices = ['1', '2', '3', '4']

    for i in range(len(type_tegel)):
        row = 0
        for j in normalsups:
            for k in indices:
                AX1 = tl.draw_tile_symbolic(AX1, bricks, type_tegel[i], k, j, 'n',
                                   tablespacing * (i),
                                   - tablespacing * float(row + 8),
                                   True, fontsize,
                                   LINEWIDTH, CORNERSIZE, MIDDLESIZE, 
                                   SUP_SIZE, CIRCLE_SIZE, INDENTSIZE, 
                                   FACTOR, FUDGE_FACTOR_2, prototile=False, full=True)
                row = row + 1

    type_tegel = ['d']
    normalsups = ['5']
    indices = ['1', '2', '3', '4']

    for i in range(len(type_tegel)):
        row = 0
        for j in normalsups:
            for k in indices:
                AX1 = tl.draw_tile_symbolic(AX1, bricks, type_tegel[i], k, j, 'n',
                                   tablespacing * (i),
                                   - tablespacing * float(row + 12),
                                   True, fontsize,
                                   LINEWIDTH, CORNERSIZE, MIDDLESIZE, 
                                   SUP_SIZE, CIRCLE_SIZE, INDENTSIZE, 
                                   FACTOR, FUDGE_FACTOR_2, prototile=False, full=True)
                row = row + 1

    AX1.set_xticklabels([])
    AX1.set_yticklabels([])
    plt.axis('off')
    AX1.grid(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    for item in [FIG1, AX1]:
        item.patch.set_visible(False)

    AX1.set_xlim([-bumper / 2.,
                 (figwidth - 1.0) * tablespacing + 1.0 + bumper / 2. ])
    AX1.set_ylim([-bumper / 2. - (figheight - 1.0)  * tablespacing - 1.0,
                 bumper / 2.])

    AX1.set_autoscale_on(False)

    plt.gca().set_aspect(1, adjustable='box')

    plt.savefig('tabLe_symbolic.pdf', format="pdf",
                transparent=True, bbox_inches=0.0, pad_inches=0.0)
    plt.close()

    ##########################################################################
    ##########################################################################

    plotsettings = deepcopy(bricks)

    relabel_inv = {'1': '0',
                   '2': '7',
                   '3': '3',
                   '4': '5',
                   '5': '1',
                   '6': '2',
                   '7': '4',
                   '8': '6'}

    #global settings
    VERTEXSIZE = 20.0
    LINEWIDTH_VERTEX = 0.25
    LINEWIDTH = 0.15
    TABLESPACING = 3.0

    YSTRETCH = 1.2

    BUMPER = 3

    m = 10
    n = 16

    FONTSIZE = 2.5

    ANGLES = np.array([60.0, 90.0, 135.0, 75.0])

    # Default settings for plotting the meshes
    for LETTER in plotsettings.keys():
        for INDEX in plotsettings[LETTER]:
            for SUP in plotsettings[LETTER][INDEX]:
                plotsettings[LETTER][INDEX][SUP] = {}
                plotsettings[LETTER][INDEX][SUP]['xoffset_letter'] = 0.4
                plotsettings[LETTER][INDEX][SUP]['yoffset_letter'] = -0.75
                plotsettings[LETTER][INDEX][SUP]['length_top'] = 1.0
                plotsettings[LETTER][INDEX][SUP]['length_left'] = 1.0
                plotsettings[LETTER][INDEX][SUP]['dangling_ends'] = 0.5
                plotsettings[LETTER][INDEX][SUP]['global_x_shift'] = 0.0
                plotsettings[LETTER][INDEX][SUP]['global_y_shift'] = 0.0

    #Adjusting individual meshes

    plotsettings['f']['2'][relabel_inv['5']]['length_left'] = 1.4

    plotsettings['i']['1'][relabel_inv['5']]['length_top'] = 0.75
    plotsettings['i']['1'][relabel_inv['5']]['length_left'] = 1.4

    plotsettings['j']['1'][relabel_inv['5']]['length_top'] = 0.75
    plotsettings['j']['1'][relabel_inv['5']]['length_left'] = 1.4

    plotsettings['k']['1'][relabel_inv['5']]['length_top'] = 0.75
    plotsettings['k']['1'][relabel_inv['5']]['length_left'] = 1.4

    plotsettings['e']['2'][relabel_inv['5']]['length_top'] = 0.8
    plotsettings['e']['2'][relabel_inv['5']]['length_left'] = 1.2

    plotsettings['c']['3'][relabel_inv['3']]['length_top'] = 1.5
    plotsettings['c']['3'][relabel_inv['3']]['length_left'] = 0.8
    plotsettings['c']['3'][relabel_inv['3']]['dangling_ends'] = 0.4
    plotsettings['c']['3'][relabel_inv['3']]['global_x_shift'] = -0.3

    plotsettings['c']['3'][relabel_inv['6']]['length_top'] = 0.8
    plotsettings['c']['3'][relabel_inv['6']]['length_left'] = 0.8
    plotsettings['c']['3'][relabel_inv['6']]['global_x_shift'] = 0.4
    plotsettings['c']['3'][relabel_inv['6']]['dangling_ends'] = 0.4

    plotsettings['d']['3'][relabel_inv['6']]['length_top'] = 0.8
    plotsettings['d']['3'][relabel_inv['6']]['length_left'] = 0.8
    plotsettings['d']['3'][relabel_inv['6']]['global_x_shift'] = 0.4
    plotsettings['d']['3'][relabel_inv['6']]['dangling_ends']   = 0.4

    plotsettings['e']['3'][relabel_inv['6']]['length_top'] = 0.6
    plotsettings['e']['3'][relabel_inv['6']]['length_left'] = 1.2

    plotsettings['f']['3'][relabel_inv['6']]['length_top'] = 0.6
    plotsettings['f']['3'][relabel_inv['6']]['length_left'] = 1.35
    plotsettings['f']['3'][relabel_inv['6']]['dangling_ends'] = 0.4

    plotsettings['g']['3'][relabel_inv['6']]['length_top'] = 0.6
    plotsettings['g']['3'][relabel_inv['6']]['length_left'] = 1.35

    plotsettings['h']['3'][relabel_inv['6']]['length_top'] = 0.6
    plotsettings['h']['3'][relabel_inv['6']]['length_left'] = 1.35

    plotsettings['j']['3'][relabel_inv['6']]['length_top'] = 0.6
    plotsettings['j']['3'][relabel_inv['6']]['length_left'] = 1.4

    plotsettings['c']['1'][relabel_inv['7']]['length_top'] = 0.7
    plotsettings['c']['1'][relabel_inv['7']]['length_left'] = 1.3

    plotsettings['e']['4'][relabel_inv['7']]['length_top'] = 0.75
    plotsettings['f']['4'][relabel_inv['7']]['length_top'] = 0.75
    plotsettings['g']['4'][relabel_inv['7']]['length_top'] = 0.75
    plotsettings['h']['4'][relabel_inv['7']]['length_top'] = 0.75

    plotsettings['d']['3'][relabel_inv['3']]['length_top'] = 1.4
    plotsettings['d']['3'][relabel_inv['3']]['length_left'] = 0.6

    plotsettings['d']['1'][relabel_inv['4']]['length_top'] = 1.4
    plotsettings['d']['1'][relabel_inv['4']]['length_left'] = 0.8

    plotsettings['d']['3'][relabel_inv['8']]['length_top'] = 0.75
    plotsettings['d']['3'][relabel_inv['8']]['length_left'] = 1.5

    plotsettings['d']['4'][relabel_inv['4']]['length_top'] = 1.5
    plotsettings['d']['4'][relabel_inv['4']]['length_left'] = 0.9

    plotsettings['h']['2'][relabel_inv['8']]['length_top'] = 0.85
    plotsettings['h']['2'][relabel_inv['8']]['length_left'] = 1.0
    plotsettings['h']['2'][relabel_inv['8']]['dangling_ends'] = 0.4

    plotsettings['i']['2'][relabel_inv['8']]['length_top'] = 0.7
    plotsettings['i']['2'][relabel_inv['8']]['length_left'] = 1

    plotsettings['j']['2'][relabel_inv['8']]['length_top'] = 0.7
    plotsettings['j']['2'][relabel_inv['8']]['length_left'] = 1

    plotsettings['k']['2'][relabel_inv['8']]['length_top'] = 0.7
    plotsettings['k']['2'][relabel_inv['8']]['length_left'] = 1

    plotsettings['e']['3'][relabel_inv['8']]['length_top'] = 0.7
    plotsettings['e']['3'][relabel_inv['8']]['length_left'] = 1.0
    plotsettings['e']['3'][relabel_inv['8']]['dangling_ends'] = 0.4

    plotsettings['f']['3'][relabel_inv['8']]['length_top'] = 0.7
    plotsettings['f']['3'][relabel_inv['8']]['length_left'] = 1.0
    plotsettings['f']['3'][relabel_inv['8']]['dangling_ends'] = 0.4

    plotsettings['g']['3'][relabel_inv['8']]['length_top'] = 0.7
    plotsettings['g']['3'][relabel_inv['8']]['length_left'] = 1.0
    plotsettings['g']['3'][relabel_inv['8']]['dangling_ends'] = 0.4

    plotsettings['h']['3'][relabel_inv['8']]['length_top'] = 0.7
    plotsettings['h']['3'][relabel_inv['8']]['length_left'] = 1.0
    plotsettings['h']['3'][relabel_inv['8']]['dangling_ends'] = 0.4

    plotsettings['e']['2'][relabel_inv['8']]['length_left'] = 1.3
    plotsettings['f']['2'][relabel_inv['8']]['length_left'] = 1.3
    plotsettings['g']['2'][relabel_inv['8']]['length_left'] = 1.3
    plotsettings['h']['2'][relabel_inv['8']]['length_left'] = 1.3

    plotsettings['j']['2'][relabel_inv['7']]['length_left'] = 1.3

    plotsettings['e']['1'][relabel_inv['7']]['length_left'] = 1.3

    FIG = plt.figure(figsize=(float(m) * factor_canvas + bumper,
                              float(n) * YSTRETCH * factor_canvas + bumper),
                     frameon=False)

    AX = FIG.add_axes([0, 0, 1, 1], aspect='equal')
    AX.set_xticklabels([])
    AX.set_yticklabels([])
    plt.axis('off')
    AX.grid(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    for item in [FIG, AX]:
        item.patch.set_visible(False)


    type_tegel = ['d', 'e', 'f', 'g', 'h']
    normalsups = ['1', '2', '4', '6']
    indices = ['1', '2', '3', '4']
    AX = plot_mesh(AX, type_tegel, normalsups, indices, [1, 2, 1, 0, 1, 0],
                   plotsettings,bricks,ANGLES, TABLESPACING, FONTSIZE,
                   LINEWIDTH, VERTEXSIZE, LINEWIDTH_VERTEX)

    type_tegel = ['i', 'k']
    normalsups = ['1', '2', '4', '6']
    indices = ['1', '2']
    AX = plot_mesh(AX, type_tegel, normalsups, indices, [2, 7, 1, 0, 1, 2],
                   plotsettings,bricks,ANGLES, TABLESPACING, FONTSIZE,
                   LINEWIDTH, VERTEXSIZE, LINEWIDTH_VERTEX)

    type_tegel = ['j']
    normalsups = ['1', '2', '4', '6']
    indices = ['1', '2', '3', '4']
    AX = plot_mesh(AX, type_tegel, normalsups, indices, [1, 8, 1, 0, 1, 0],
                   plotsettings,bricks,ANGLES, TABLESPACING, FONTSIZE,
                   LINEWIDTH, VERTEXSIZE, LINEWIDTH_VERTEX)

    type_tegel = ['c']
    normalsups = ['1', '2', '4']
    indices = ['1', '2', '3', '4']
    AX = plot_mesh(AX, type_tegel, normalsups, indices, [1, 1, 1, 0, 1, 0],
                   plotsettings,bricks,ANGLES, TABLESPACING, FONTSIZE,
                   LINEWIDTH, VERTEXSIZE, LINEWIDTH_VERTEX)

    type_tegel = ['a']
    normalsups = ['0', '7']
    indices = ['1']
    AX = plot_mesh(AX, type_tegel, normalsups, indices, [1, 0, 2, 0, 1, 0],
                   plotsettings,bricks,ANGLES, TABLESPACING, FONTSIZE,
                   LINEWIDTH, VERTEXSIZE, LINEWIDTH_VERTEX)

    type_tegel = ['b']
    normalsups = ['0', '7']
    indices = ['1']
    AX = plot_mesh(AX, type_tegel, normalsups, indices, [1, 0, 2, 1, 1, 0],
                   plotsettings,bricks,ANGLES, TABLESPACING, FONTSIZE,
                   LINEWIDTH, VERTEXSIZE, LINEWIDTH_VERTEX)

    type_tegel = ['c']
    normalsups = ['3']
    indices = ['1', '2', '3', '4']
    AX = plot_mesh(AX, type_tegel, normalsups, indices, [1, 0, 1, 4, 1, 0],
                   plotsettings,bricks,ANGLES, TABLESPACING, FONTSIZE,
                   LINEWIDTH, VERTEXSIZE, LINEWIDTH_VERTEX)

    type_tegel = ['d']
    normalsups = ['3']
    indices = ['1', '2', '3', '4']
    AX = plot_mesh(AX, type_tegel, normalsups, indices, [1, 0, 1, 8, 1, 0],
                   plotsettings,bricks,ANGLES, TABLESPACING, FONTSIZE,
                   LINEWIDTH, VERTEXSIZE, LINEWIDTH_VERTEX)

    type_tegel = ['d']
    normalsups = ['5']
    indices = ['1', '2', '3', '4']
    AX = plot_mesh(AX, type_tegel, normalsups, indices, [1, 0, 1, 12, 1, 0],
                   plotsettings,bricks,ANGLES, TABLESPACING, FONTSIZE,
                   LINEWIDTH, VERTEXSIZE, LINEWIDTH_VERTEX)

    print 'done w/ plotting'

    AX.set_xlim([-BUMPER / 2.0,
                 (float(m) - 1) * TABLESPACING + 1. + BUMPER / 2.0])

    AX.set_ylim([-BUMPER / 2.0 -
                 TABLESPACING * YSTRETCH * (float(n) - 1) - 1 * YSTRETCH,
                 BUMPER / 2.0])

    AX.set_autoscale_on(False)

    plt.gca().set_aspect(1, adjustable='box')
    plt.savefig('table_real_space.pdf', format="pdf",
                transparent=True, bbox_inches=0.0, pad_inches=0.0)
    plt.close()
