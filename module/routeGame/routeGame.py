import numpy as np
import matplotlib.pyplot as plt

class routeGame:
    def __init__(self, depots, nodes, seed=1, geo=''):
        np.random.seed(seed)
        if type(depots) != int:
            raise ValueError('Depots must be int')
        if type(nodes) != int:
            raise ValueError('Nodes must be int')

        self.__geo=geo
        self.__depots = depots
        self.__deliveries = nodes

        self.__coordinates = self.__coords(depots, nodes)
        self.__indices = self.__index_labels(depots, nodes)
        self.__distance_matrix = self.__cal_distance_matrix(self.__coordinates)

    def __repr__(self):
        return "Generates synthetic routing assignment"

    def __index_labels(self, depots, nodes):
        indices = {}
        for i in range(depots + nodes):
            if i < depots:
                indices[i] = {'type': 'depot'}
            else:
                indices[i] = {'type': 'delivery'}
        return indices

    def __coords(self, depots, nodes):
        if self.__geo == 'circle':
            coo = []
            for i in range(depots + nodes):
               coo.append([  np.random.rand() + np.cos(i*2.0*np.pi/(depots + nodes)),    np.random.rand() +np.sin(i*2.0*np.pi/(depots + nodes))])
            return np.asarray(coo)
        else:
            coo = np.random.random([depots+nodes, 2])
            return coo

    def __cal_distance_matrix(self, coords):
        distance_matrix = np.zeros([len(coords), len(coords)])
        for idx, c in enumerate(coords):
            for idy, q in enumerate(coords):
                distance_matrix[idx, idy] = self.__travel_time(idx, idy, coords)
        return distance_matrix

    def __travel_time(self, i, j, coords, indices=False):

        if i < len(coords) and j < len(coords):
            start_x = coords[i, 0]
            start_y = coords[i, 1]

            end_x = coords[j, 0]
            end_y = coords[j, 1]

            distance = pow(pow(start_x - end_x, 2) + pow(start_y - end_y, 2), 0.5)
        else:
            raise IndexError("Index out of bounds")

        return distance

    def viz(self):
        colors = np.random.rand(len( self.__coordinates))
        plt.scatter(self.__coordinates[:, 0], self.__coordinates[:, 1], s=15, c=colors, alpha=0.5)
        plt.show()

    @property
    def depots(self):
        return self.__depots

    @property
    def deliveries(self):
        return self.__deliveries

    @property
    def index(self):
        return self.__indices

    @property
    def coordinates(self):
        return self.__coordinates

    @property
    def cost_matrix(self):
        return self.__distance_matrix

    @property
    def time_matrix(self):
        return self.__distance_matrix*3

    @property
    def type(self):
        return 'TSPTW'
