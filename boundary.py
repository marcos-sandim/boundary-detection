# coding: utf-8

# The MIT License (MIT)
#
# Copyright (c) 2016 Marcos Sandim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys
import os
import math
import time
import numpy as np
import configparser
from sklearn.neighbors import BallTree as NeighborsTree
from scipy.spatial import ConvexHull


class DatReader:
    """Helper class used to read the data from text files.

    Helper class used to read the data from text files.


    Attributes:
        data_dir: Directory with the data files.
        base_name: Base name (prefix) for the data files.
        extension: Extension of the data files.
        current_step: Number of the step being processed.
        steps: Array containing the numbers of the steps waiting to be processed.
    """

    def __init__(self, config, config_file):
        """Constructor

        Initializes the reader with the parameters provided by the configuration file.

        Args:
            config: The ConfigParser object.
            config_file: The path of the config file being used.
        """
        self.data_dir = config.get('data', 'directory', fallback='./')
        self.base_name = config.get('data', 'basename', fallback='pdata.')
        self.extension = config.get('data', 'extension', fallback='.dat')
        self.dimension = config.getint('data', 'dimension', fallback=3)

        self.current_step = None
        self.steps = []

        if (self.data_dir in ['./', '', '.\\', '.']):
            self.data_dir = os.path.dirname(config_file)
        self.data_dir = os.path.join(self.data_dir, '')

        if (config.getboolean('steps', 'linear', fallback=True)):
            for step in range(config.getint('steps', 'start', fallback=1),
                              config.getint('steps', 'end', fallback=1) + 1):
                self.steps.append(step)
        else:
            steps = config.get('steps', 'steps', fallback='1')
            for step in steps.split():
                self.steps.append(int(step))

    def has_next(self):
        """Checks if there still are steps to be processed.

        Checks if there still are steps to be processed.

        Returns:
            True if the steps array is not empty, False otherwise.
        """
        return len(self.steps) > 0

    def get_next_step(self):
        """Returns the next step to be processed.

        Returns the next step to be processed and pops its number from the steps array.

        Returns:
            A numpy array containing the coordinates of the particles,
            or None if there are no more steps to be processed.
        """
        if (not self.has_next()):
            return None

        self.current_step = self.steps.pop(0)

        input_fpath = '%s%s%d%s' % (self.data_dir, self.base_name,
                                    self.current_step, self.extension)

        return np.loadtxt(input_fpath, usecols=to_index_tuple(np.arange(self.dimension)))


class SupportGrid:
    """Grid structure to support the computation of viewpoints.

    Grid structure to support the computation of viewpoints that will be used
    to detect the rho-boundary of a particle system which particle's positions
    are stored in the 'points' array.

    Attributes:
        points: A numpy array containing the position of the particles.
        rho: The value of rho, in general the h value from SPH simulations
            is a good approximation.
        dimension: The dimension of the particle system and the grid.
        cell_size: The length of the cells edges.
        aabb_min: The lower corner of the Axis Aligned Bounding Box containing
            the points.
        aabb_max: The upper corner of the Axis Aligned Bounding Box containing
            the points.
        grid_dims: The number of cells along each axis needed to compute the
            viewpoints, it includes some padding cells on each side.
        grid_min: The lower corner of the grid.
        grid_max: The upper corner of the grid.
        grid_count: A numpy array used to keep the number of points per cell.
        grid_elems: A numpy array containing lists of the indexes of the points
            inside each cell.
        tree: A KDTree structure used to simplify and speedup neighborhood queries.
        neighbor_cell_list: A numpy array with indexes in {-1, 0, 1} used to
            assist the traversal of neighboring cells in any dimension >= 1.
    """

    def __init__(self, points, rho, dimension):
        """Constructor

        Initializes the grid and helper structures using the provided points
        and rho parameter.

        Args:
            points: A numpy array containing the coordinates of the particles.
            rho: Needed to compute the rho-boundary of the system.
            dimension: The dimension of the particle system.
        """
        self.points = points
        self.rho = rho
        self.dimension = dimension
        self.cell_size = 2.0 * rho

        self.aabb_min = np.amin(points, axis=0)
        self.aabb_max = np.amax(points, axis=0)

        self.grid_dims = (self.aabb_max - self.aabb_min) / self.cell_size
        # Regarding the + 3: 1 for left side, 1 for right side, 1 for rounding
        # up
        self.grid_dims = np.trunc(self.grid_dims) + 3
        self.grid_dims = self.grid_dims.astype(int)

        self.grid_min = self.aabb_min - self.cell_size
        self.grid_max = self.grid_min + self.grid_dims * self.cell_size

        self.grid_count = np.zeros(self.grid_dims, dtype=int)
        self.grid_elems = np.empty(self.grid_dims, dtype=object)

        self.update_grid()
        self.tree = NeighborsTree(
            self.points, leaf_size=10, metric='euclidean')

        self.neighbor_cell_list = self.compute_neighbor_cell_list()

    def update_grid(self):
        """Updates the grid with the counting and indexes.

        Updates the grid with the number of particles in each cell and puts
        the index of each particle in the corresponding cell.
        """
        for i in range(self.points.shape[0]):
            pt = self.points[i]

            idx = (pt - self.grid_min) / self.cell_size
            idx = to_index_tuple(idx)
            self.grid_count[idx] += 1
            if (self.grid_elems[idx] == None):
                self.grid_elems[idx] = []
            self.grid_elems[idx].append(i)

    def compute_neighbor_cell_list(self):
        """Computes a list of offsets to the neighboring cells.

        Computes a list of offsets to the neighboring cells based on the
        dimension. This is used to simplify the traversal of neighbor cells in
        any dimension. For a 2D grid it produces:
        [[-1 -1], [-1 0], [-1 1], [0 -1], [0 0], [0 1], [1 -1], [1 0], [1 1]].
        By using this list we can visit all the 9 cells around a point or cell
        with a single loop.

        Returns:
            A numpy array containing a list of offests to neighboring cells.
        """
        previous = np.array([[-1], [0], [1]], dtype=int)
        current = None
        current_n_rows = 3
        for c in range(1, self.dimension):
            ones = np.ones((current_n_rows, 1))
            for i in range(-1, 2):
                temp = np.hstack((ones * i, previous))
                if (current is None):
                    current = temp
                else:
                    current = np.vstack((current, temp))

            current_n_rows *= 3
            previous = current
            current = None

        return previous

    def get_viewpoints(self):
        """Computes and returns the viewpoints that will be used by the instances
        of the HPR operator.

        Computes and returns the viewpoints that will be used by the instances
        of the HPR operator. Empty cells neighboring non-empty cells get a
        viewpoint in its center; Non-empty cells that have no empty neighbor go
        through an additional step to generate viewpoints in cavity cells.

        Returns:
            A numpy array containing the viewpoints.
        """
        self.viewpoints = []

        # for i in range(self.grid_dims[0]):
        #    for j in range(self.grid_dims[1]):
        #        for k in range(self.grid_dims[2]):

        for cell in range(self.grid_dims.prod()):
            idx = np.unravel_index(cell, self.grid_dims)
            if (self.grid_count[idx] == 0):
                self.process_empty_cell(idx)
            else:
                self.process_nonempty_cell(idx)

        return self.viewpoints

    def process_empty_cell(self, idx):
        """Processes an empty cell and produces a viewpoint on its center.

        Processes an empty cell and produces a viewpoint on its center.
        The viewpoint is created only if the empty cell has a non-empty neighbor
        cell.

        Args:
            idx: The index of the cell.
        """
        for i in range(self.neighbor_cell_list.shape[0]):
            n_idx = idx + self.neighbor_cell_list[i]

            # check grid limits
            if (np.any(np.less(n_idx, np.zeros([1, self.dimension]))) or
                    np.any(np.greater_equal(n_idx, self.grid_dims))):
                continue

            n_idx = to_index_tuple(n_idx)

            # If there is a nonempty neighbor, we place a viewpoint
            # at the center of the current cell
            if (self.grid_count[n_idx] != 0):
                viewpoint = self.grid_min + \
                    np.array(idx) * self.cell_size + 0.5 * self.cell_size
                self.viewpoints.append(viewpoint)
                return

    def process_nonempty_cell(self, idx):
        """Processes an non-empty cell and produces viewpoints if possible.

        Processes an non-empty cell and produces a set of viewpoints based on the
        points inside the cell and its distribution.

        Args:
            idx: The index of the cell.
        """
        # Check if there is an empty neighbor,
        # in this case the empty neighbor should be enough
        for i in range(self.neighbor_cell_list.shape[0]):
            n_idx = idx + self.neighbor_cell_list[i]

            # check grid limits
            if (np.any(np.less(n_idx, np.zeros([1, self.dimension]))) or
                    np.any(np.greater_equal(n_idx, self.grid_dims))):
                continue

            n_idx = to_index_tuple(n_idx)

            if (self.grid_count[n_idx] == 0):
                return

        # Get everyone in the cell, and define a new viewpoint candidate,
        # based on its neighborhood centroid
        for i in range(self.grid_count[idx]):
            ii = self.grid_elems[idx][i]

            pt = self.points[ii]

            neighbors = self.tree.query_radius(pt.reshape(1,-1), r=2.0 * self.rho)[0]

            centroid = np.sum(
                self.points[neighbors], axis=0) / neighbors.shape[0]

            V = pt - centroid
            V = V / np.linalg.norm(V)

            viewpoint = pt + V * self.rho

            neighbors = self.tree.query_radius(viewpoint.reshape(1,-1), r=0.95 * self.rho)[0]
            if (neighbors.size == 0):
                self.viewpoints.append(viewpoint)

    def get_candidates(self, viewpoint):
        """Gets a set of points that are candidates to be marked as boundary.

        Gets a set of points that are candidates to be marked as boundary. These
        candidates are inside the local neighbohood of a viewpoint and will be
        used on the HPR operator.

        Args:
            viewpoint: The viewpoint that will be used by the HPR operator.

        Returns:
            A numpy array containing the boundary candidates around the viewpoint.
        """
        return self.tree.query_radius(viewpoint.reshape(1,-1), r=4.0 * self.rho)[0]


def to_index_tuple(idx):
    """Converts a numpy array to a tuple of integer indexes.

    Converts a numpy array to a tuple of integer indexes.

    Args:
        idx: The numpy array containing the indexes.

    Returns:
        A tuple of indexes.
    """
    return tuple(idx.astype(int).tolist())


def exponential_flip(points, viewpoint, gamma):
    """Performs exponential flip

    Performs exponential flip/transform on a set of points w.r.t to a viewpoint.
    The the viewpoint is moved to the origin, along with the points, scaled to
    fit into the unit circle and then flipped.

    Args:
        points: The numpy array of points to be flipped.
        viewpoint: The viewpoint to be used as the center of the flipping.
        gamma: The exponent used to flip the points.

    Returns:
        A numpy array containing the flipped points.
    """
    # Define a scale factor to fit everything into the unit circle,
    # with the viewpoint at the center
    maxnorm = 0.0
    for i in range(points.shape[0]):
        pi = points[i] - viewpoint
        normp = np.linalg.norm(pi)
        maxnorm = normp if normp > maxnorm else maxnorm

    # With some gap
    maxnorm *= 1.1

    # Transform/flip each point w.r.t the viewpoint
    flipped = np.zeros(points.shape)
    for i in range(points.shape[0]):
        pi = points[i] - viewpoint
        pi = pi / maxnorm
        normp = np.linalg.norm(pi)

        flipped[i] = pi / math.pow(normp, gamma)

    return flipped


def run_hpr(grid, viewpoint, gamma, boundary):
    """Runs the HPR operator for a single viewpoint and its neighborhood

    Runs the HPR operator for a single viewpoint and its neighborhood.

    Args:
        grid: The support grid containing the data
        viewpoint: The viewpoint to be used by the HPR operator.
        gamma: The exponent used to flip the points.
        boundary: An array of flags used as output.
    """
    candidates = grid.get_candidates(viewpoint)

    if (candidates.shape[0] == 0):
        return

    if (candidates.shape[0] <= grid.dimension):
        boundary[candidates] = True
        return

    flipped = exponential_flip(grid.points[candidates], viewpoint, gamma)

    # add the viewpoint to the end of the list
    flipped = np.vstack([flipped, np.zeros([1, grid.dimension])])

    hull = ConvexHull(flipped)
    visible_idx = hull.vertices

    # remove the index corresponding to the viewpoint
    visible_idx.sort()
    visible_idx = np.delete(visible_idx, -1)

    visible_idx = candidates[visible_idx]

    boundary[visible_idx] = True


if (len(sys.argv) < 2):
    print('usage: python %s config_file.ini' % sys.argv[0])
    sys.exit(-1)

config_file = sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)

reader = DatReader(config, config_file)

rho = config.getfloat('boundary', 'rho', fallback=1.0)
gamma = config.getfloat('hpr', 'gamma', fallback=2.0)

output_dir = reader.data_dir + '/output_' + time.strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(output_dir, '')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while (reader.has_next()):
    points = reader.get_next_step()

    grid = SupportGrid(points, rho, reader.dimension)

    viewpoints = grid.get_viewpoints()

    boundary = np.zeros([points.shape[0], 1], dtype=np.bool_)

    for vp in viewpoints:
        run_hpr(grid, vp, gamma, boundary)

    output_fpath = '%s%s%d.out' % (output_dir,
                                   reader.base_name, reader.current_step)
    np.savetxt(output_fpath, boundary, fmt='%d')
