import numpy as np
import matplotlib.pyplot as plt

class RouteProblem:
    def __init__(self, depots, nodes):
        if type(depots) != int:
            raise ValueError('Depots must be int')
        if type(nodes) != int:
            raise ValueError('Nodes must be int')

        self.__coordinates = self.__coords(depots, nodes)
        self.__indices = self.__index_labels(depots, nodes)
        self.__distance_matrix = self.__cal_distance_matrix(self.__coordinates)

    def __repr__(self):
        return "Generates synthetic routing assignment"

    def __index_labels(self, depots, nodes):
        indices = {}
        for i in range(depots + nodes):
            if i < depots:
                indices[i] = 'depot'
            else:
                indices[i] = 'delivery'
        return indices

    def __coords(self, depots, nodes):
        return np.random.random([depots+nodes, 2])

    def __cal_distance_matrix(self, coords):
        distance_matrix = np.zeros([len(coords), len(coords)])
        for idx, c in enumerate(coords):
            for idy, q in enumerate(coords):
                distance_matrix[idx, idy] = self.__travel_time(idx, idy, coords)
        return distance_matrix

    def __travel_time(self, i, j, coords, indices=False):
        distance = 0

        if i < len(coords) and j < len(coords):
            start_x = coords[i, 0]
            start_y = coords[i, 0]

            end_x = coords[j, 0]
            end_y = coords[j, 0]

            distance = pow(pow(start_x - end_x, 2) + pow(start_y - end_y, 2), 0.5)
        else:
            raise IndexError("Index out of bounds")

        return distance

    def viz(self):
        colors = np.random.rand(len( self.__coordinates))
        plt.scatter(self.__coordinates[:, 0], self.__coordinates[:, 1], s=15, c=colors, alpha=0.5)
        plt.show()

    @property
    def index(self):
        return self.__indices

    @property
    def coordinates(self):
        return self.__coordinates

    @property
    def distance_matrix(self):
        return self.__distance_matrix

gameInstance = RouteProblem(2, 10)
gameInstance.viz()
