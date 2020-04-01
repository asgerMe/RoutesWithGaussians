from module.routeGame import routeGame as game
import numpy as np
import matplotlib.pyplot as plt
import itertools

class Sample:

    def __init__(self, ASSIGNMENT, samples=1000, burns=500, chains=4, step_size=0.3):

        self.__check_assignment(ASSIGNMENT)

        self.num_results = samples
        self.num_burnin_steps = burns
        self.num_chains = chains
        self.step_size = step_size

        self.transition_matrix = self.__init_weights(self.distance_matrix)
        #Non essential - Ditch on implementation
        self.coordinates = ASSIGNMENT.coordinates

        self.a = 1

        self.ROUTE = [[0], 999999]

    def __check_assignment(self, ASSIGNMENT):
        try:
            self.type = ASSIGNMENT.type
        except AttributeError:
           raise AttributeError('The Assignment type should be specified. List of objectives is: {TSP, TSPTW}')

        try:
            self.index = ASSIGNMENT.index
            self.depots = 0
            self.nodes = 0

            for v in self.index.values():
                if v == 'depot':
                    self.depots += 1
                if v == 'delivery':
                    self.nodes += 1

        except AttributeError:
           raise AttributeError('The Assignment index should be specified. A dictionary with depot, or node label for each entry ')

        try:
            self.distance_matrix = ASSIGNMENT.distance_matrix
        except AttributeError:
           raise AttributeError('The Assignment distance_matrix should be specified.')

        if self.type == 'TSPTW':
            try:
                self.time_windows = ASSIGNMENT.time_windows
            except AttributeError:
                raise AttributeError('For TSPTW problems, the time_window attribute should be specified.')

            try:
                self.time_matrix = ASSIGNMENT.time_matrix
            except AttributeError:
                raise AttributeError('For TSPTW problems, the time_matrix should be specified.')

    def __call__(self, ce_samples=100, batch=15, quantile=0.01):
        y = 9999999

        for _ in range(ce_samples):
            sri, sl = self.__get_route_batch(batch)
            sl_idx = np.argsort(sl)
            sl = sl[sl_idx]

            sort_sri = []
            for i in range(len(sl_idx)):
                sort_sri.append(sri[sl_idx[i]][:])
            quantile_val = np.quantile(sl, quantile)

            if quantile_val < y:
                y = quantile_val

            sl_max = sl[sl <= y]
            sri_max = sort_sri[0:len(sl_max)][:]
            Is = np.sum(sl_max)

            if Is > 0:
                print(self.ROUTE[1])
                if self.ROUTE[1] > sl_max[0]:
                    self.ROUTE = [sri_max[0], sl_max[0]]

                old_r = 0
                for idx, R in enumerate(sri_max):
                    for r in R:
                        if old_r != r:
                            self.transition_matrix[old_r, r] += self.a*sl_max[idx]/Is
                        old_r = r

        self.ROUTE[0] = self.__MCMC_sample()
        return 'finished'

    def __MCMC_sample(self, size=5):
        if True:
            print('init untangling')
            mc_score = 0
            best_route_copy = np.copy(self.ROUTE[0])
            for idx, idxr in enumerate(best_route_copy):
                best_route = self.MC_untangle(best_route_copy, best_route_copy[idx], size=size)
                if idx < len(best_route_copy)-1:
                    mc_score += self.distance_matrix[best_route[idx], best_route[idx+1]]
            print('After untangeling:', mc_score)

        return best_route

    def MC_untangle(self, route_assignment, start, size=3):
        size += 2
        end = start + size
        start = np.clip(start, 1, len(route_assignment)-1)
        end = np.clip(end, 1, len(route_assignment)-1)

        partial_route = route_assignment[start:end]
        permutes = list(itertools.permutations(partial_route[1:-1]))

        score_list = []
        ops=0
        for perm in permutes:
            score = 0
            proposal_partial_route = np.asarray(partial_route)
            proposal_partial_route[1:-1] = np.asarray(perm)

            for idx, val in enumerate(proposal_partial_route):
                ops += 1
                if idx < len(proposal_partial_route)-1:
                    score += self.distance_matrix[proposal_partial_route[idx], proposal_partial_route[idx+1]]
            score_list.append(score)

        idx_min = np.argmin(score_list)
        best_perm = np.asarray(permutes[idx_min])
        partial_route[1:-1] = best_perm

        route_assignment[start:end] = partial_route

        return route_assignment

    def __get_route(self):
        tm = np.copy(self.transition_matrix)
        indexes = list(range(self.nodes + 1))

        row_idx = 0
        store_row_idx = [0]
        score_list = [0]

        for _ in range(self.nodes):
            trans_probs = np.copy(tm[:, row_idx])
            tm[row_idx, :] *= 0
            tm[:, row_idx] *= 0

            norm_probs = np.sum(trans_probs)
            probs = trans_probs/norm_probs
            transition = np.random.choice(indexes, size=1, p=probs)[0]

            score = self.distance_matrix[row_idx, transition]
            score_list.append(score)

            store_row_idx.append(transition)
            row_idx = transition

        return store_row_idx, score_list

    def __get_route_batch(self, batch=1):
        route_batch = []
        score_batch = []

        for _ in range(batch):
            route, score = self.__get_route()
            route_batch.append(route)
            score_batch.append(score)

        return route_batch, np.sum(score_batch, axis=1)


    def __init_weights(self, distance_matrix, distance_weights=True):
        transition_matrix = np.ones(np.shape(distance_matrix)) - np.eye(np.shape(distance_matrix)[0])
        transition_matrix /= (np.shape(distance_matrix)[0] - 1)

        if distance_weights:
            for i in range(np.shape(distance_matrix)[0]):
                for j in range(np.shape(distance_matrix)[1]):
                    transition_matrix[i, j] = np.exp(-20*distance_matrix[i, j])

            transition_matrix -= np.eye(np.shape(transition_matrix)[0])
            for i in range(np.shape(distance_matrix)[0]):
                transition_matrix[i, :] = transition_matrix[i, :] / np.sum(transition_matrix[i, :])

        return transition_matrix

    def display_route(self):
        print(self.ROUTE[1])
        self.coordinates = self.coordinates[self.ROUTE[0]]
        colors = np.random.rand(len(self.coordinates))
        plt.plot(self.coordinates[:, 0], self.coordinates[:, 1], alpha=0.35)
        plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], s=40, c=colors, alpha=0.85)
        for i in range(len(self.coordinates[:, 0])):
            plt.text(self.coordinates[i, 0], self.coordinates[i, 1], str(i))
        plt.show()


game = game.routeGame(1, 30, seed=20)
ps = Sample(game)
ps(batch=20, ce_samples=500)
ps.display_route()





