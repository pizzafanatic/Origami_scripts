import numpy as np
import xlsxwriter
from matplotlib import pyplot as plt
from copy import deepcopy


def draw_tile_symbolic(ax, bricks, bricktype, index, sup_index,\
        orientation, xpos, ypos, letter, font_size, linewidth, cornersize,\
        middlesize, sup_size, circle_size, indentsize,\
        prototile=False, full=False):

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

    colors_angles = ['dodgerblue', 'limegreen', 'orange', 'red']
    colors_sup = ['w', 'w', 'w', 'w', 'gray', 'gray', 'gray', 'gray']

    name = bricktype

    if not prototile:
        if bricktype != 'A' and bricktype != 'B':
            name = name + '_{' + index + '}'
        else:
            name = name + '_{ }'
        name = name + '^' + relabel[sup_index]
    else:
        if bricktype != 'A' and bricktype != 'B':
            name = name + '_{' + index + '}'
        else:
            name = name + '_{ }'
        name = name + '^{ }'

    brick_mod = np.mod(brick - 1, 4)

    if (letter is True):
        ax.text(0.5 + xpos, -0.5 + ypos, r'$' + name + '$',
                fontsize=font_size,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=90.0 * convert_directions[orientation])

    if (full is True):
        line_elements_x = [[0.0 + xpos, cornersize + xpos],
                           [0.0 + xpos, 0.0 + xpos],
                           [0 + xpos, 0 + xpos],
                           [0 + xpos, cornersize + xpos],
                           [1 - cornersize + xpos, 1 + xpos],
                           [1 + xpos, 1 + xpos],
                           [1 + xpos, 1 + xpos],
                           [1 + xpos, 1 - cornersize + xpos],
                           [0.5 - middlesize + xpos, 0.5 + middlesize + xpos],
                           [0.5 - middlesize + xpos, 0.5 + middlesize + xpos],
                           [0 + xpos, 0 + xpos],
                           [1 + xpos, 1 + xpos]]

        line_elements_y = [[0.0 + ypos, 0.0 + ypos],
                           [0.0 + ypos, - cornersize + ypos],
                           [cornersize - 1 + ypos, - 1 + ypos],
                           [-1 + ypos, - 1 + ypos],
                           [0 + ypos, 0 + ypos],
                           [0 + ypos, - cornersize + ypos],
                           [cornersize - 1 + ypos, - 1 + ypos],
                           [-1 + ypos, - 1 + ypos],
                           [0 + ypos, 0 + ypos],
                           [- 1 + ypos, - 1 + ypos],
                           [- 0.5 + middlesize + ypos,
                            - 0.5 - middlesize + ypos],
                           [- 0.5 + middlesize + ypos,
                            - 0.5 - middlesize + ypos]]

        for i in range(len(line_elements_x)):
            ax.plot(line_elements_x[i], line_elements_y[i],
                    c='k',
                    lw=linewidth,
                    solid_capstyle='round',
                    zorder=2)

        quarter_circle_x = circle_size \
            - circlefier(0.5 * np.pi, 2 * circle_size)[0] + xpos
        quarter_circle_y = - circlefier(0.5 * np.pi, 2 * circle_size)[1] + ypos
        quarter_circle_x[1] = quarter_circle_x[0]
        quarter_circle_y[1] = quarter_circle_y[0]
        quarter_circle_x[0] = 0 + xpos
        quarter_circle_y[0] = 0 + ypos
        v1 = plt.Polygon(np.transpose(np.array([quarter_circle_x,
                                                quarter_circle_y])),
                         color=colors_angles[np.int(brick_mod[1, 1])],
                         linewidth=0)
        # plt.gca().add_patch(v1)

        ax.add_patch(v1)


        quarter_circle_x = sup_size \
            - circlefier(0.5 * np.pi, 2 * sup_size)[0] + xpos
        quarter_circle_y = - circlefier(0.5 * np.pi, 2 * sup_size)[1] + ypos
        quarter_circle_x[1] = quarter_circle_x[0]
        quarter_circle_y[1] = quarter_circle_y[0]
        quarter_circle_x[0] = 0 + xpos
        quarter_circle_y[0] = 0 + ypos

        v11 = plt.Polygon(np.transpose(np.array([quarter_circle_x,
                                                quarter_circle_y])),
                          color=colors_sup[brick[1, 1] - 1],
                          linewidth=linewidth,
                          zorder=1)
        ax.add_patch(v11)

        quarter_circle_x = 1 - circle_size \
            + circlefier(0.5 * np.pi, 2 * circle_size)[0] + xpos
        quarter_circle_y = - circlefier(0.5 * np.pi, 2 * circle_size)[1] + ypos
        quarter_circle_x[1] = quarter_circle_x[0]
        quarter_circle_y[1] = quarter_circle_y[0]
        quarter_circle_x[0] = 1 + xpos
        quarter_circle_y[0] = 0 + ypos

        v2 = plt.Polygon(np.transpose(np.array([quarter_circle_x,
                                                quarter_circle_y])),
                         color=colors_angles[brick_mod[1, 2]],
                         linewidth=0)
        ax.add_patch(v2)

        quarter_circle_x = 1 - sup_size \
            + circlefier(0.5 * np.pi, 2 * sup_size)[0] + xpos
        quarter_circle_y = - circlefier(0.5 * np.pi, 2 * sup_size)[1] + ypos
        quarter_circle_x[1] = quarter_circle_x[0]
        quarter_circle_y[1] = quarter_circle_y[0]
        quarter_circle_x[0] = 1.0 + xpos
        quarter_circle_y[0] = ypos

        v22 = plt.Polygon(np.transpose(np.array([quarter_circle_x,
                                                 quarter_circle_y])),
                          color=colors_sup[brick[1, 2] - 1],
                          linewidth=linewidth,
                          zorder=1)
        ax.add_patch(v22)

        quarter_circle_x = 1 - circle_size \
            + circlefier(0.5 * np.pi, 2 * circle_size)[0] + xpos
        quarter_circle_y = -1 + \
            circlefier(0.5 * np.pi, 2 * circle_size)[1] + ypos
        quarter_circle_x[1] = quarter_circle_x[0]
        quarter_circle_y[1] = quarter_circle_y[0]
        quarter_circle_x[0] = 1 + xpos
        quarter_circle_y[0] = -1 + ypos
        v3 = plt.Polygon(np.transpose(np.array([quarter_circle_x,
                                                quarter_circle_y])),
                         color=colors_angles[brick_mod[2, 2]],
                         linewidth=0)
        ax.add_patch(v3)

        quarter_circle_x = 1 - sup_size \
            + circlefier(0.5 * np.pi, 2 * sup_size)[0] + xpos
        quarter_circle_y = -1 +\
            circlefier(0.5 * np.pi, 2 * sup_size)[1] + ypos
        quarter_circle_x[1] = quarter_circle_x[0]
        quarter_circle_y[1] = quarter_circle_y[0]
        quarter_circle_x[0] = 1.0 + xpos
        quarter_circle_y[0] = - 1.0 + ypos

        v33 = plt.Polygon(np.transpose(np.array([quarter_circle_x,
                                                 quarter_circle_y])),
                          color=colors_sup[brick[2, 2] - 1],
                          linewidth=linewidth,
                          zorder=1)
        ax.add_patch(v33)

        quarter_circle_x = circle_size -\
            circlefier(0.5 * np.pi, 2 * circle_size)[0] + xpos

        quarter_circle_y = - 1 + circlefier(0.5 * np.pi,
                                            2 * circle_size)[1] + ypos
        quarter_circle_x[1] = quarter_circle_x[0]
        quarter_circle_y[1] = quarter_circle_y[0]
        quarter_circle_x[0] = xpos
        quarter_circle_y[0] = - 1 + ypos

        v4 = plt.Polygon(np.transpose(np.array([quarter_circle_x,
                                                quarter_circle_y])),
                         color=colors_angles[brick_mod[2, 1]],
                         linewidth=0)
        ax.add_patch(v4)

        quarter_circle_x = sup_size \
            - circlefier(0.5 * np.pi, 2 * sup_size)[0] + xpos
        quarter_circle_y = -1 +\
            circlefier(0.5 * np.pi, 2 * sup_size)[1] + ypos
        quarter_circle_x[1] = quarter_circle_x[0]
        quarter_circle_y[1] = quarter_circle_y[0]
        quarter_circle_x[0] = - 0.0 + xpos
        quarter_circle_y[0] = - 1.0 + ypos

        v44 = plt.Polygon(np.transpose(np.array([quarter_circle_x,
                                                 quarter_circle_y])),
                          color=colors_sup[brick[2, 1] - 1],
                          linewidth= linewidth,
                          zorder=1)
        ax.add_patch(v44)

        midpoints = np.array([[cornersize + 0.5 * indentsize, 0],
                              [1 - cornersize - 0.5 * indentsize, 0],
                              [1, - cornersize - 0.5 * indentsize],
                              [1, - 1 + cornersize + 0.5 * indentsize],
                              [1 - cornersize - 0.5 * indentsize, -1],
                              [cornersize + 0.5 * indentsize, -1],
                              [0, -1 + cornersize + 0.5 * indentsize],
                              [0, -cornersize - 0.5 * indentsize]])

        square_up = np.array([[- 0.5 * indentsize, 0],
                              [- 0.5 * indentsize, 0.6 * indentsize],
                              [+ 0.5 * indentsize, 0.6 * indentsize],
                              [+ 0.5 * indentsize, 0]])

        circle_up = np.transpose(np.array([-0.5 * indentsize +
                                 circlefier(np.pi, indentsize)[0],
                                 circlefier(np.pi, indentsize)[1]]))

        triangle_60_up = np.array([[- 0.5 * indentsize, 0],
                                   [0, 0.4 * indentsize * 3 ** 0.5],
                                   [+ 0.5 * indentsize, 0]])

        circle_up_right = np.transpose(np.array(
            [circlefier(0.5 * np.pi,
                        indentsize)[0],
             - (- 0.5 * indentsize +
             circlefier(0.5 * np.pi, indentsize)[1])]))

        circle_up_left = deepcopy(circle_up_right)
        circle_up_left[:, 0] = - circle_up_left[:, 0]

        convex = np.vstack([circle_up_right, circle_up_left])

        convex_sort = convex[convex[:, 0].argsort()]

        convex_sort[:, 1] = 1.5 * convex_sort[:, 1]

        shapes_N = [convex_sort,
                    triangle_60_up,
                    circle_up,
                    square_up]

        shapes_W, shapes_S, shapes_E = [], [], []

        for i in range(4):
            shapes_E.append(rotatexy(shapes_N[i], 90))
            shapes_S.append(rotatexy(shapes_N[i], 180))
            shapes_W.append(rotatexy(shapes_N[i], 270))

        if((brick_mod[1, 1] - brick_mod[0, 1]) == - 1 or
           (brick_mod[1, 1] - brick_mod[0, 1]) == 3):
            ax.plot(midpoints[0][0] + xpos + shapes_N[brick_mod[1, 1]][:, 0],
                    midpoints[0][1] + ypos + shapes_N[brick_mod[1, 1]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[1, 1] - brick_mod[0, 1]) == 1 or
           (brick_mod[1, 1] - brick_mod[0, 1]) == - 3):
            ax.plot(midpoints[0][0] + xpos + shapes_S[brick_mod[0, 1]][:, 0],
                    midpoints[0][1] + ypos + shapes_S[brick_mod[0, 1]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[1, 1] - brick_mod[0, 1]) == - 1 or
           (brick_mod[1, 1] - brick_mod[0, 1]) == 3):
            ax.plot(midpoints[7][0] + xpos + shapes_E[brick_mod[1, 0]][:, 0],
                    midpoints[7][1] + ypos + shapes_E[brick_mod[1, 0]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[1, 1] - brick_mod[0, 1]) == 1 or
           (brick_mod[1, 1] - brick_mod[0, 1]) == - 3):
            ax.plot(midpoints[7][0] + xpos + shapes_W[brick_mod[1, 1]][:, 0],
                    midpoints[7][1] + ypos + shapes_W[brick_mod[1, 1]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[1, 2] - brick_mod[0, 2]) == - 1 or
           (brick_mod[1, 2] - brick_mod[0, 2]) == 3):
            ax.plot(midpoints[1][0] + xpos + shapes_N[brick_mod[1, 2]][:, 0],
                    midpoints[1][1] + ypos + shapes_N[brick_mod[1, 2]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[1, 2] - brick_mod[0, 2]) == 1 or
           (brick_mod[1, 2] - brick_mod[0, 2]) == - 3):
            ax.plot(midpoints[1][0] + xpos + shapes_S[brick_mod[0, 2]][:, 0],
                    midpoints[1][1] + ypos + shapes_S[brick_mod[0, 2]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[1, 2] - brick_mod[0, 2]) == - 1 or
           (brick_mod[1, 2] - brick_mod[0, 2]) == 3):
            ax.plot(midpoints[2][0] + xpos + shapes_W[brick_mod[1, 3]][:, 0],
                    midpoints[2][1] + ypos + shapes_W[brick_mod[1, 3]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[1, 2] - brick_mod[0, 2]) == 1 or
           (brick_mod[1, 2] - brick_mod[0, 2]) == - 3):
            ax.plot(midpoints[2][0] + xpos + shapes_E[brick_mod[1, 2]][:, 0],
                    midpoints[2][1] + ypos + shapes_E[brick_mod[1, 2]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[2, 2] - brick_mod[2, 3]) == - 1 or
           (brick_mod[2, 2] - brick_mod[2, 3]) == 3):
            ax.plot(midpoints[3][0] + xpos + shapes_E[brick_mod[2, 2]][:, 0],
                    midpoints[3][1] + ypos + shapes_E[brick_mod[2, 2]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[2, 2] - brick_mod[2, 3]) == 1 or
           (brick_mod[2, 2] - brick_mod[2, 3]) == - 3):
            ax.plot(midpoints[3][0] + xpos + shapes_W[brick_mod[2, 3]][:, 0],
                    midpoints[3][1] + ypos + shapes_W[brick_mod[2, 3]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[2, 2] - brick_mod[2, 3]) == - 1 or
           (brick_mod[2, 2] - brick_mod[2, 3]) == 3):
            ax.plot(midpoints[4][0] + xpos + shapes_N[brick_mod[3, 2]][:, 0],
                    midpoints[4][1] + ypos + shapes_N[brick_mod[3, 2]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[2, 2] - brick_mod[2, 3]) == 1 or
           (brick_mod[2, 2] - brick_mod[2, 3]) == - 3):
            ax.plot(midpoints[4][0] + xpos + shapes_S[brick_mod[2, 2]][:, 0],
                    midpoints[4][1] + ypos + shapes_S[brick_mod[2, 2]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[2, 1] - brick_mod[2, 0]) == - 1 or
           (brick_mod[2, 1] - brick_mod[2, 0]) == 3):
            ax.plot(midpoints[5][0] + xpos + shapes_N[brick_mod[3, 1]][:, 0],
                    midpoints[5][1] + ypos + shapes_N[brick_mod[3, 1]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[2, 1] - brick_mod[2, 0]) == 1 or
           (brick_mod[2, 1] - brick_mod[2, 0]) == - 3):
            ax.plot(midpoints[5][0] + xpos + shapes_S[brick_mod[2, 1]][:, 0],
                    midpoints[5][1] + ypos + shapes_S[brick_mod[2, 1]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

        if((brick_mod[2, 1] - brick_mod[2, 0]) == - 1 or
           (brick_mod[2, 1] - brick_mod[2, 0]) == 3):
            ax.plot(midpoints[6][0] + xpos + shapes_W[brick_mod[2, 1]][:, 0],
                    midpoints[6][1] + ypos + shapes_W[brick_mod[2, 1]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')
        if((brick_mod[2, 1] - brick_mod[2, 0]) == 1 or
           (brick_mod[2, 1] - brick_mod[2, 0]) == - 3):
            ax.plot(midpoints[6][0] + xpos + shapes_E[brick_mod[2, 0]][:, 0],
                    midpoints[6][1] + ypos + shapes_E[brick_mod[2, 0]][:, 1],
                    c='k', lw=linewidth, solid_capstyle='round')

    return ax


def plot_tiling(ax, coords, raw_tiling, name, 
                linewidth, vertexsize, linewidth_vertex):
    m = coords.shape[0]
    n = coords.shape[1]

    for i in range(m - 1):
        for j in range(n - 2):
            ax.plot(np.array([coords[i, j + 1, 0],
                               coords[i + 1, j + 1, 0]]),
                     np.array([coords[i, j + 1, 1], coords[i + 1, j + 1, 1]]),
                     c='k', lw=linewidth, zorder=0)

    for j in range(n - 1):
        for i in range(m - 2):
            ax.plot(np.array([coords[i + 1, j, 0], coords[i + 1, j + 1, 0]]),
                     np.array([coords[i + 1, j, 1], coords[i + 1, j + 1, 1]]),
                     c='k', lw=linewidth, zorder=0)
    k = m * n

    for i in range(m - 2):
        for j in range(n - 2):
            if(4 * i == raw_tiling.shape[0]):
                vertex_x = 4 * i - 1
            else:
                vertex_x = 4 * i
            if(4 * j == raw_tiling.shape[1]):
                vertex_y = 4 * j - 1
            else:
                vertex_y = 4 * j
            VERTEX = raw_tiling[vertex_x, vertex_y]

            if(VERTEX < 5):
                ax.scatter(coords[i + 1, j + 1, 0], coords[i + 1, j + 1, 1],
                            c='w', edgecolor='k', s=vertexsize / float(k),
                            marker='o', linewidth=linewidth_vertex, zorder=2)

            if(VERTEX > 4):
                ax.scatter(coords[i + 1, j + 1, 0], coords[i + 1, j + 1, 1],
                            c='gray', edgecolor='k', s=vertexsize / float(k),
                            marker='o', linewidth=linewidth_vertex, zorder=2)
    return ax   

def cyclifier(proto, cyclic_permutations, i):
    # this function generates cyclic permutations
    proto = 10 * proto
    if(i == 0):
        return proto / 10
    elif(cyclic_permutations == 2 and i == 1):
        proto[np.where(proto == 40)] = 1
        proto[np.where(proto == 30)] = 4
        proto[np.where(proto == 20)] = 3
        proto[np.where(proto == 10)] = 2
        return proto
    elif(cyclic_permutations == 4 and i == 1):
        proto[np.where(proto == 40)] = 1
        proto[np.where(proto == 30)] = 4
        proto[np.where(proto == 20)] = 3
        proto[np.where(proto == 10)] = 2
        return proto
    elif(cyclic_permutations == 4 and i == 2):
        proto[np.where(proto == 40)] = 2
        proto[np.where(proto == 20)] = 4
        proto[np.where(proto == 10)] = 3
        proto[np.where(proto == 30)] = 1
        return proto
    elif(cyclic_permutations == 4 and i == 3):
        proto[np.where(proto == 40)] = 3
        proto[np.where(proto == 30)] = 2
        proto[np.where(proto == 20)] = 1
        proto[np.where(proto == 10)] = 4
        return proto
    else:
        raise Exception('dit kan niet')
        return '?????'

def circlefier(theta, diameter):
    points = np.linspace(0, theta, 100)
    x = - 0.5 * diameter * np.cos(points) + 0.5 * diameter
    y = 0.5 * diameter * np.sin(points)
    return x, y


def rotatexy(X, angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c, - s], [s, c]])
    return X * R


def rot_vector(tile, orientation):
    if(orientation == 'n'):
        return tile
    elif(orientation == 'w'):
        return np.rot90(tile, 1)
    elif(orientation == 's'):
        return np.rot90(tile, 2)
    elif(orientation == 'e'):
        return np.rot90(tile, 3)


def extract_tiling(tiling):
    m, n = tiling[1].shape[0], tiling[1].shape[1]
    raw = tiling[0]
    kolom1 = 4 * np.arange(0, n - 1) + 4
    kolom2 = kolom1 + 1
    deletee = np.hstack([kolom1, kolom2])
    raw = np.delete(raw, deletee, axis=1)
    rij1 = 4 * np.arange(0, m - 1) + 4
    rij2 = rij1 + 1
    deletee = np.hstack([rij1, rij2])
    raw = np.delete(raw, deletee, axis=0)
    return m, n, raw

def rotate_and_rescale(A, B, angle, new_length):
    # B is the vector that we rotate, around point
    # A
    # with angle angle (in radians, ccw)
    # returns the x,y coordinates of rotated vector B
    norm_AB = np.linalg.norm(B)
    rotatee = np.zeros([2])
    C = np.zeros([2])
    rotated = np.zeros([2])
    rotatee[0] = B[0]
    rotatee[1] = B[1]
    rotated[0] = np.cos(angle) * rotatee[0] - np.sin(angle) * rotatee[1]
    rotated[1] = np.sin(angle) * rotatee[0] + np.cos(angle) * rotatee[1]
    scale_factor = new_length / norm_AB
    C = A + scale_factor * rotated
    return C


def output_csv_coords(coords, Z, factor=1.0):
    no_edges = coords[1:-1, 1:-1, :]

    m = no_edges.shape[0]
    n = no_edges.shape[1]

    X_coords = no_edges[:, :, 0] * factor
    Y_coords = no_edges[:, :, 1] * factor

    X_coords = X_coords.reshape((m * n, 1))
    Y_coords = Y_coords.reshape((m * n, 1))

    workbook = xlsxwriter.Workbook(Z + '.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, 'mm')
    worksheet.write(1, 0, 'x')
    worksheet.write(1, 1, 'y')

    for i in range(m * n):
        worksheet.write(2 + i, 0, X_coords[i])
        worksheet.write(2 + i, 1, Y_coords[i])


def draw_tiling(tiling, angles_4, top_l, left_l, sidelength):
    angles = np.hstack([angles_4, np.pi - angles_4])
    m, n, tiling = extract_tiling(tiling)
    top_l_2 = np.hstack([np.array([sidelength]),
                         top_l, np.array([sidelength])])
    left_l_2 = np.hstack([np.array([sidelength]),
                          left_l, np.array([sidelength])])
    m = 3 + m
    n = 3 + n

    coords = np.zeros([m, n, 2])
    coords[:] = np.nan
    coords[1, 1, 0] = 0.0
    coords[1, 1, 1] = 0.0
    coords[1, 2, 0] = top_l_2[1]
    coords[1, 2, 1] = 0.0
    coords[2, 1] = rotate_and_rescale(coords[1, 1], coords[1, 2],
                                      -angles[tiling[1, 1] - 1],
                                      left_l_2[1])
    alpha = angles[tiling[0, 1] - 1]
    beta = angles[tiling[0, 0] - 1]
    coords[0, 1] = rotate_and_rescale(coords[1, 1, :], coords[1, 2, :],
                                      alpha, sidelength)
    coords[1, 0] = rotate_and_rescale(coords[1, 1, :], coords[1, 2, :],
                                      alpha + beta, sidelength)

    i = 1
    for j in range(n - 3):
        rotation = angles[tiling[i, 2 * j + 2] - 1] +\
            angles[tiling[i, 2 * j + 3] - 1]
        locatie = coords[i, j + 2]
        vector_uit_oorsprong = coords[i, j + 1] - coords[i, j + 2]
        coords[i, j + 3] = rotate_and_rescale(locatie, vector_uit_oorsprong,
                                              rotation, top_l_2[j + 2])

    i = 0
    for j in range(n - 3):
        rotation = angles[tiling[i, 2 * j + 2] - 1]
        locatie = coords[i + 1, j + 2]
        vector_uit_oorsprong = coords[i + 1, j + 1] - coords[i + 1, j + 2]
        coords[i, j + 2] = rotate_and_rescale(locatie, vector_uit_oorsprong,
                                              -rotation, sidelength)

    j = 1
    for i in range(m - 3):
        rotation = angles[tiling[2 * i + 2, j] - 1] +\
            angles[tiling[2 * i + 3, j] - 1]
        locatie = coords[i + 2, j]
        vector_uit_oorsprong = coords[i + 1, j] - coords[i + 2, j]
        coords[i + 3, j] = rotate_and_rescale(locatie, vector_uit_oorsprong,
                                              - rotation, left_l_2[i + 2])

    j = 0
    for i in range(m - 3):
        rotation = angles[tiling[2 * i + 2, j] - 1]
        locatie = coords[i + 2, j + 1]
        vector_uit_oorsprong = coords[i + 1, j + 1] - coords[i + 2, j + 1]
        coords[i + 2, j] = rotate_and_rescale(locatie,
                                              vector_uit_oorsprong,
                                              rotation,
                                              sidelength)

    # A B
    # C D
    for i in range(m - 3):
        for j in range(n - 3):
            A = coords[i + 1, j + 1]
            B0 = coords[i + 1, j + 2]
            C0 = coords[i + 2, j + 1]
            AC0 = A - C0
            AB0 = A - B0
            beta = angles[tiling[2 * i + 1, 2 * j + 2] - 1]
            gamma = angles[tiling[2 * i + 2, 2 * j + 1] - 1]
            B1 = rotate_and_rescale(B0, AB0, beta, 1.0)
            C1 = rotate_and_rescale(C0, AC0, -gamma, 1.0)
            t, s = np.linalg.solve(np.array([B1 - B0, C0 - C1]).T, C0 - B0)
            coords[i + 2, j + 2, :] = (1 - t) * B0 + t * B1

    for i in range(m - 3):
        rotation = angles[tiling[2 * i + 2, -1] - 1]
        locatie = coords[i + 2, -2]
        vector_uit_oorsprong = coords[i + 1, - 2] - coords[i + 2, -2]
        coords[i + 2, -1] = rotate_and_rescale(locatie,
                                               vector_uit_oorsprong,
                                               -rotation,
                                               sidelength)

    for j in range(n - 3):
        rotation = angles[tiling[-1, 2 * j + 2] - 1]
        locatie = coords[-2, j + 2]
        vector_uit_oorsprong = coords[-2, j + 1] - coords[-2, j + 2]
        coords[-1, j + 2] = rotate_and_rescale(locatie, 
                                               vector_uit_oorsprong,
                                               rotation, sidelength)
    return coords




sups = [np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])]

# sups.append()

sups.append(np.array([[4, 4, 0, 0],
                      [4, 4, 0, 0],
                      [0, 0, 4, 4],
                      [0, 0, 4, 4]]))

sups.append(np.array([[0, 0, 4, 4],
                      [0, 0, 4, 4],
                      [4, 4, 0, 0],
                      [4, 4, 0, 0]]))

sups.append(np.array([[4, 4, 4, 4],
                      [4, 4, 4, 4],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]))

sups.append(np.array([[0, 0, 4, 4],
                      [0, 0, 4, 4],
                      [0, 0, 4, 4],
                      [0, 0, 4, 4]]))

sups.append(np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [4, 4, 4, 4],
                      [4, 4, 4, 4]]))

sups.append(np.array([[4, 4, 0, 0],
                      [4, 4, 0, 0],
                      [4, 4, 0, 0],
                      [4, 4, 0, 0]]))

sups.append(np.array([[4, 4, 4, 4],
                      [4, 4, 4, 4],
                      [4, 4, 4, 4],
                      [4, 4, 4, 4]]))

verts = [np.array([[1, 2], [4, 3]]), np.array([[2, 3], [1, 4]]),
         np.array([[3, 4], [2, 1]]), np.array([[4, 1], [3, 2]]),
         np.array([[4, 3], [1, 2]]), np.array([[1, 4], [2, 3]]),
         np.array([[2, 1], [3, 4]]), np.array([[3, 2], [4, 1]])]


tile = {}

tile['a'] = np.empty([4, 4], dtype='int')

tile['a'][0:2, 0:2] = verts[7]  # topleft
tile['a'][0:2, 2:4] = verts[5]  # topright
tile['a'][2:4, 0:2] = verts[5]  # bottomleft
tile['a'][2:4, 2:4] = verts[7]  # bottomright

tile['b'] = np.empty([4, 4], dtype='int')

tile['b'][0:2, 0:2] = verts[2]  # topleft
tile['b'][0:2, 2:4] = verts[0]  # topright
tile['b'][2:4, 0:2] = verts[0]  # bottomleft
tile['b'][2:4, 2:4] = verts[2]  # bottomright

tile['c'] = np.empty([4, 4], dtype='int')

tile['c'][0:2, 0:2] = verts[2]  # topleft
tile['c'][0:2, 2:4] = verts[4]  # topright
tile['c'][2:4, 0:2] = verts[6]  # bottomleft
tile['c'][2:4, 2:4] = verts[0]  # bottomright

tile['d'] = np.empty([4, 4], dtype='int')

tile['d'][0:2, 0:2] = verts[7]  # topleft
tile['d'][0:2, 2:4] = verts[1]  # topright
tile['d'][2:4, 0:2] = verts[6]  # bottomleft
tile['d'][2:4, 2:4] = verts[0]  # bottomright

tile['e'] = np.empty([4, 4], dtype='int')

tile['e'][0:2, 0:2] = verts[3]  # topleft
tile['e'][0:2, 2:4] = verts[5]  # topright
tile['e'][2:4, 0:2] = verts[6]  # bottomleft
tile['e'][2:4, 2:4] = verts[0]  # bottomright

tile['f'] = np.empty([4, 4], dtype='int')

tile['f'][0:2, 0:2] = verts[4]  # topleft
tile['f'][0:2, 2:4] = verts[2]  # topright
tile['f'][2:4, 0:2] = verts[6]  # bottomleft
tile['f'][2:4, 2:4] = verts[0]  # bottomright

tile['g'] = np.empty([4, 4], dtype='int')

tile['g'][0:2, 0:2] = verts[3]  # topleft
tile['g'][0:2, 2:4] = verts[5]  # opright
tile['g'][2:4, 0:2] = verts[3]  # bottomleft
tile['g'][2:4, 2:4] = verts[5]  # bottomright

tile['h'] = np.empty([4, 4], dtype='int')

tile['h'][0:2, 0:2] = verts[4]  # topleft
tile['h'][0:2, 2:4] = verts[2]  # topright
tile['h'][2:4, 0:2] = verts[3]  # bottomleft
tile['h'][2:4, 2:4] = verts[5]  # bottomright

tile['i'] = np.empty([4, 4], dtype='int')

tile['i'][0:2, 0:2] = verts[0]  # topleft
tile['i'][0:2, 2:4] = verts[6]  # topright
tile['i'][2:4, 0:2] = verts[6]  # bottomleft
tile['i'][2:4, 2:4] = verts[0]  # bottomright

tile['j'] = np.empty([4, 4], dtype='int')

tile['j'][0:2, 0:2] = verts[5]  # topleft
tile['j'][0:2, 2:4] = verts[3]  # topright
tile['j'][2:4, 0:2] = verts[6]  # bottomleft
tile['j'][2:4, 2:4] = verts[0]  # bottomright

tile['k'] = np.empty([4, 4], dtype='int')

tile['k'][0:2, 0:2] = verts[5]  # topleft
tile['k'][0:2, 2:4] = verts[3]  # topright
tile['k'][2:4, 0:2] = verts[3]  # bottomleft
tile['k'][2:4, 2:4] = verts[5]  # bottomright

tiles = {}
tiles['a'] = {}
tiles['a']['tile'] = tile['a']
tiles['a']['sup_patterns'] = [0, 7]
tiles['a']['cyclic'] = 1

tiles['b'] = {}
tiles['b']['tile'] = tile['b']
tiles['b']['sup_patterns'] = [0, 7]
tiles['b']['cyclic'] = 1

tiles['c'] = {}
tiles['c']['tile'] = tile['c']
tiles['c']['sup_patterns'] = [1, 2, 3, 4]
tiles['c']['cyclic'] = 4

tiles['d'] = {}
tiles['d']['tile'] = tile['d']
tiles['d']['sup_patterns'] = [1, 2, 3, 4, 5, 6]
tiles['d']['cyclic'] = 4

tiles['e'] = {}
tiles['e']['tile'] = tile['e']
tiles['e']['sup_patterns'] = [1, 2, 4, 6]
tiles['e']['cyclic'] = 4

tiles['f'] = {}
tiles['f']['tile'] = tile['f']
tiles['f']['sup_patterns'] = [1, 2, 4, 6]
tiles['f']['cyclic'] = 4

tiles['g'] = {}
tiles['g']['tile'] = tile['g']
tiles['g']['sup_patterns'] = [1, 2, 4, 6]
tiles['g']['cyclic'] = 4

tiles['h'] = {}
tiles['h']['tile'] = tile['h']
tiles['h']['sup_patterns'] = [1, 2, 4, 6]
tiles['h']['cyclic'] = 4

tiles['i'] = {}
tiles['i']['tile'] = tile['i']
tiles['i']['sup_patterns'] = [1, 2, 4, 6]
tiles['i']['cyclic'] = 2

tiles['j'] = {}
tiles['j']['tile'] = tile['j']
tiles['j']['sup_patterns'] = [1, 2, 4, 6]
tiles['j']['cyclic'] = 4

tiles['k'] = {}
tiles['k']['tile'] = tile['k']
tiles['k']['sup_patterns'] = [1, 2, 4, 6]
tiles['k']['cyclic'] = 2


tiles_34 = {}
bricks = {}


for tegel in tile.keys():
    bricks[tegel] = {}
    tiles_34[tegel] = {}
    for i in range(tiles[tegel]['cyclic']):
        prototile = cyclifier(tiles[tegel]['tile'],
                              tiles[tegel]['cyclic'], i)
        tiles_34[tegel][str(i + 1)] = prototile
        bricks[tegel][str(i + 1)] = {}
        for j in tiles[tegel]['sup_patterns']:
            bricks[tegel][str(i + 1)][str(j)] = prototile + sups[j]


convert_directions = {}
convert_directions = {0: 'n', 1: 'w', 2: 's', 3: 'e'}

monster = np.empty([140, 4, 4, 4])
convert_cd = {}
convert_dc = {}

j = 0
for brick_type in bricks:
    convert_cd[brick_type] = {}
    for brick_index in bricks[brick_type]:
        convert_cd[brick_type][brick_index] = {}
        for supp in bricks[brick_type][brick_index]:
            convert_cd[brick_type][brick_index][supp] = j
            for i in range(4):
                monster[j, i, :, :] =\
                    np.rot90(bricks[brick_type][brick_index][supp], k=i)
            j = j + 1

for brick_type in convert_cd:
    for brick_index in convert_cd[brick_type]:
        for supp in convert_cd[brick_type][brick_index]:
            convert_dc[convert_cd[brick_type][brick_index][supp]] = \
                brick_type + brick_index + supp


if __name__ == "__main__":
    print('main')

