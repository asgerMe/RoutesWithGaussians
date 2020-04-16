import pickle
from scipy.spatial import KDTree
from firebase_admin import credentials, firestore, initialize_app
import numpy as np
import matplotlib.pyplot as plt
import ee
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from nested_lookup import nested_lookup
import imageio
import seaborn as sns
sns.set_style("white")

class BayesianInterpolator:

    def __init__(self, uid, keyPath='', update_pc=False):

        ee.Initialize()
        dataset = ee.ImageCollection('NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4').filter(ee.Filter.date('2010-01-01', '2010-01-31'))
        self.__image = ee.Image(dataset.select('avg_vis').first())
        self.__update = update_pc

        if update_pc:
            cred = credentials.Certificate(keyPath)
            self.__default_app = initialize_app(cred)
            self.__db = firestore.client()

        self.__point_cloud = {}
        self.__coordinates = []
        self.__values = []

        self.__kd_tree = {}
        self.__kd_tree_euclidian = {}

        self.__load_data()
        self.__create_kd_trees()

    def __repr__(self):
        return 'Queue new data points, fetch proxy data and vizualize results !'

    def __call__(self):
        return 0

    def __look_up_data(self, N, all_features=True, seed=1002):
        np.random.seed(seed)
        r_idx = np.random.randint(low=0, high=np.shape(self.coordinates)[0])
        random_coord = self.coordinates[r_idx, :]
        v, idx = self.query_coordinates(random_coord, k=N, viz=False)

        idx_t = list(idx)
        if all_features:
            v_l, idx_l = self.query_distance(random_coord, k=N, viz=False)
            idx_t = np.concatenate([idx, idx_l], axis=0)
            idx_t = list(set(idx_t))

        idx_t.remove(r_idx)

        x = self.coordinates[idx_t]
        lp = self.__get_pixel_values(x)
        lp = lp / np.max(lp)
        x = x / np.max(x)

        y = self.values[idx_t, 0]
        y_mean_prior = self.values[idx_t, 0]
        y_std_prior = np.square(y_mean_prior / 3)

        y_mean_predictive = np.mean(y_mean_prior)
        y_std_predictive = np.var(y_mean_prior)

        y_mean_prior = list(y_mean_prior)
        y_std_prior = list(y_std_prior)

        y_mean_prior.append(y_mean_predictive)
        y_std_prior.append(y_std_predictive)
        l = np.sqrt(np.square(x[:, 0] - x[:, 2]) + np.square(x[:, 1] - x[:, 3]))
        new_data = dict(N=len(x), x=x, y=y, l=l, lp=lp)
        return new_data

    def hyper_parameter_inference(self, N0=500, samples=1000, chains=2, warmup=500, upsamples=3):
        sigma = []
        alpha = []
        alpha_l = []
        alpha_lp = []
        rho = []
        rho_l = []
        rho_lp = []
        N = []

        new_data = self.__look_up_data(N=N0, all_features=False, seed=1200)
        sm = pickle.load(open('./Models/CompiledModels/prior.pkl', 'rb'))
        for s in range(upsamples):
            pick_samples = (1+s)*N0/upsamples
            idx = np.random.randint(low=0, high=len(new_data['x']), size=int(pick_samples))
            new_data['N'] = len(idx)
            new_data['x'] = new_data['x'][idx]
            new_data['y'] = new_data['y'][idx]
            new_data['l'] = new_data['l'][idx]
            new_data['lp'] = new_data['lp'][idx]

            fit = sm.sampling(data=new_data, iter=samples, chains=chains, warmup=warmup)
            ex = fit.extract()

            if 'sigma' in ex:
                sigma.append(ex['sigma'])
            if 'alpha' in ex:
                alpha.append(ex['alpha'])
            if 'alpha_l' in ex:

                 alpha_l.append(ex['alpha_l'])
            if 'alpha_lp' in ex:
                alpha_lp.append(ex['alpha_lp'])
            if 'rho' in ex:
                rho.append(ex['rho'])
            if 'rho_l' in ex:
                rho_l.append(ex['rho_l'])
            if 'rho_lp' in ex:
                rho_lp.append(ex['rho_lp'])

            N.append(pick_samples)
        self.__plot_hyperparameters(sigma, rho, rho_l, rho_lp, alpha, alpha_l, alpha_lp, N)

    def __plot_hyperparameters(self, sigma, rho, rho_l, rho_lp, alpha, alpha_l, alpha_lp, N):
        if len(N) > 4:
            raise ValueError('Only four plots supported')

        titles = ['Sigma', 'Alpha', 'Alpha L', 'Alpha LP', 'Rho', 'Rho L', 'Rho LP']
        values = [sigma, alpha, alpha_l, alpha_lp, rho, rho_l, rho_lp]
        colors = [(138 / 255, 80 / 255, 179 / 255), (1, 160 / 255, 1 / 255), (89 / 255, 179 / 255, 127 / 255), (237/255, 179/255, 1)]

        fig, _ = plt.subplots(nrows=1, ncols=len(values))
        fig.patch.set_facecolor('#E0E0E0')
        fig.patch.set_alpha(0.90)

        gs1 = gridspec.GridSpec(1, len(values))
        gs1.update(wspace=0.00, hspace=0.25)  # set the spacing between axes.

        for col in range(len(values)):

            if len(values[col]) == 0:
                continue

            ax = plt.subplot(gs1[col])
            ax.yaxis.set_visible(False)
            ax.set_facecolor((0.80, 0.80, 0.80))
            ax.set_title(titles[col])
            ax.set_ylim(0, 1.2)
            for row in range(len(N)):
                sns.distplot(values[col][row], color=colors[row], kde_kws={"color": colors[row], "lw": 3, "label": str(int(N[row])) + ' interp routes', "shade": True}, ax=ax, bins=10,)

        plt.show()

    def __generate_prior_mean_and_variance(self, x):
        return 0

    def __get_pixel_values(self, coordinates):
        lines = []
        for c in coordinates:
            lines.append(ee.Feature(ee.Geometry.LineString([[c[1], c[0]], [c[3], c[2]]])))
        lines = ee.List(lines)
        pts = ee.FeatureCollection(lines)
        dict = self.__image.reduceRegions(pts, ee.Reducer.mean(), 300)

        return nested_lookup('mean', dict.getInfo())[1:]

    def __load_point_cloud_db(self):
        docs = self.__db.collection('pointcloud').stream()
        for idx, doc in enumerate(docs):
                self.__point_cloud[idx] = doc.to_dict()['data']
        return 'KD Trees loaded successfully', 200

    def __save_point_cloud(self, obj, name='POINTCLOUD'):
        with open('data/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def __load_point_cloud(self, name='POINTCLOUD'):
        with open('data/' + name + '.pkl', 'rb') as f:
            pc_format = pickle.load(f)
            self.__coordinates = []
            self.__values = []

            for pc in pc_format.values():
                pc = np.reshape(pc, [-1, 6])

                if len(self.__coordinates) == 0:
                    self.__coordinates = pc[:, :4]
                    self.__values = pc[:, 4:]

                self.__coordinates = np.concatenate([self.__coordinates, pc[:, :4]], axis=0)
                self.__values = np.concatenate([self.__values, pc[:, 4:]], axis=0)

    def __create_kd_trees(self):
        euc = np.transpose([self.__euclidian_distance()])
        self.__kd_tree = self.__init_kd_tree(self.__coordinates)
        self.__kd_tree_euclidian = self.__init_kd_tree(euc)

    def __euclidian_distance(self):
        euc = []
        for c in self.coordinates:
            r = pow(pow(c[0] - c[2], 2) + pow(c[1] - c[3], 2), 0.5)
            euc.append(r)

        return euc

    def __load_data(self):
        if self.__update:
            self.__load_point_cloud_db()

            #SAVE?LOAD - THIS SHOULD BE GOOGLE STORAGE IN THE WILD
            self.__save_point_cloud(obj=self.__point_cloud, name='POINTCLOUD')
            self.__load_point_cloud()
        else:
            self.__load_point_cloud()

    def __init_kd_tree(self, x):
        return KDTree(x)

    def create_gif(self, x, y, length = 300):
        y_sort_idx = np.argsort(y[:, -1])
        y = y[y_sort_idx, :]

        x = np.transpose(x)
        x0 = x[:, -1]

        r = pow(pow(x[0, :] - x0[0], 2) + pow(x[1, :] - x0[1], 2), 0.5) \
            + pow(pow(x[2, :] - x0[2], 2) + pow(x[3, :] - x0[3], 2), 0.5)

        idx = np.argsort(r)
        filenames = []
        for frame in range(len(y)):
            if frame > length:
                break

            plt.plot(r[idx], y[frame, :][idx], linestyle='--', color='gray')
            plt.scatter(r[idx], y[frame, :][idx], color='black', s=5)

            axes = plt.gca()
            #axes.set_ylim([-4, 1])

            file = './gif/test.' + str(frame) + '.png'
            plt.savefig(file)
            filenames.append(file)

            plt.clf()

        with imageio.get_writer('./gif/test.gif', mode='I') as writer:
            for idx in range(len(filenames)):
                filename = filenames[idx]
                image = imageio.imread(filename)
                writer.append_data(image)
        print('GIF DONE')
        return 0

    def random_lookup(self, N, euclidian=False, seed=0):
        np.random.seed(seed)
        r_idx = np.random.randint(low=0, high=np.shape(self.coordinates)[0])
        random_coord = self.coordinates[r_idx, :]

        if euclidian:
            d, idx = self.query_distance(random_coord, k=N, viz=False)
        else:
            d, idx = self.query_coordinates(random_coord, k=N, viz=False)

        route_euclidian = np.sqrt(np.power(self.coordinates[idx, 0] - random_coord[0], 2) + np.power(self.coordinates[idx, 1] - random_coord[1], 2)) \
        + np.sqrt(np.power(self.coordinates[idx, 2] - random_coord[2], 2) + np.power(self.coordinates[idx, 3] - random_coord[3], 2))

        route_euclidian = 6371 * 2 * np.pi * route_euclidian / 360

        d_route_length = np.sqrt(np.power(self.coordinates[idx, 0] - self.coordinates[idx, 2], 2) + np.power(self.coordinates[idx, 1] - self.coordinates[idx, 3], 2)) \
            - np.sqrt(np.power(random_coord[0] - random_coord[2], 2) + np.power(random_coord[1] - random_coord[3], 2))

        d_route_length *= 6371 * 2 * np.pi * d_route_length / 360

        values = self.values[idx, 0] - self.values[r_idx, 0]
        values /= 60

        return r_idx, idx, route_euclidian, values, d_route_length

    def query_coordinates(self, request, k=30, viz=False):
        v, idx = self.__kd_tree.query(request, k=k, p=2)
        if viz:
            self.viz(idx)
        return v, idx

    def query_distance(self, request, k=30, viz=False):
        random_coord_euc = [pow(pow(request[0] - request[2], 2) + pow(request[1] - request[3], 2), 0.5)]
        v, idx = self.__kd_tree_euclidian.query(random_coord_euc, k=k)
        if viz:
            self.viz(idx=idx)
        return v, idx

    def generate_interpolator_plot(self, plots=3, N=1500, euc=False):
        fig, _ = plt.subplots(nrows=2 * plots, ncols=4, sharex=True)
        fig.patch.set_facecolor('#E0E0E0')
        fig.patch.set_alpha(0.90)
        fig.text(0.05, 0.66, 'Route Location', va='center', rotation='vertical')
        fig.text(0.05, 0.33, 'Route Length', va='center', rotation='vertical')

        gs1 = gridspec.GridSpec(2 * plots, 3)
        gs1.update(wspace=0.0, hspace=0.25)  # set the spacing between axes.
        linear_idx = 0

        for q in range(2):
            for row in range(plots):
                seed = int(100 * (1 + row))
                if q == 0:
                    euc = False
                else:
                    euc = True

                idx0, idx, d, values, d_route_length = self.random_lookup(N, euclidian=euc, seed=seed)
                lp = self.__get_pixel_values(self.coordinates[idx])
                lp0 = self.__get_pixel_values([self.coordinates[idx0]])[0]

                dlp = np.asarray(lp) - np.asarray([lp0] * len(lp))

                for col in range(3):
                    ax = plt.subplot(gs1[linear_idx])
                    ax.tick_params(axis="x", direction="in")

                    if q % 2 == 0:
                        ax.set_facecolor((0.87, 0.87, 0.87))
                    else:
                        ax.set_facecolor((0.95, 0.95, 0.95))

                    linear_idx += 1
                    if col == 0:
                        if row == 0 and q == 0:
                            ax.set_title('Euclidian distance')

                        if row == plots - 1 and q == 1:
                            ax.set_xlabel('km')

                        ax.hist(d, bins=50, facecolor=(118 / 255, 255 / 255, 3 / 255), alpha=0.95, edgecolor='black',
                                linewidth=0.1)
                        ax.set_ylabel(str(int(np.round(self.values[idx0][0] / 60))) + ' min')
                    if col == 1:
                        if row == 0 and q == 0:
                            ax.set_title('Travel time difference')
                        ax.hist(values, bins=50, facecolor=(0, 229 / 255, 1), alpha=0.95, edgecolor='black',
                                linewidth=0.1)
                        ax.yaxis.set_visible(False)
                        if row == plots - 1 and q == 1:
                            ax.set_xlabel('min')

                    if col == 2:
                        if row == 0 and q == 0:
                            ax.set_title('Light pollution difference')
                        ax.hist(dlp, bins=50, facecolor=(1, 1, 0), alpha=0.95, edgecolor='black', linewidth=0.1)
                        ax.yaxis.set_visible(False)
                        if row == plots - 1 and q == 1:
                            ax.set_xlabel('NA')
        plt.show()
        return 0

    def viz(self, idx=None,   time=True, start=100, stop=5500, num=100):
        val_idx = 0
        if not time:
            val_idx = 1

        if idx is None:
            plt.hist(self.__values[:, val_idx], bins=np.linspace(start=start, stop=stop, num=num))
            plt.show()
        else:
            plt.hist(self.__values[idx, val_idx], bins=np.linspace(start=start, stop=stop, num=num))
            plt.show()

    @property
    def values(self):
        return self.__values
    @property
    def coordinates(self):
        return self.__coordinates


def main():
    MODEL = BayesianInterpolator(uid='lU6tQjFVaSgieeMaHdWzow4Ix6j1', keyPath='C:/Users/asger/Desktop/key.json')
    MODEL.hyper_parameter_inference(N0=40, samples=500, chains=2, warmup=100, upsamples=4)

if __name__ == '__main__':
    main()



