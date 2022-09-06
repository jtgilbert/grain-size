import os
import json
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample
from scipy import stats
from scipy.optimize import minimize


class GrainSize:
    """
    Estimates D50, D16, and D84 as well as the 95% confidence interval for each, for every segment of
    a drainage network using one or more field measurements of grain size
    """

    def __init__(self, network, measurements, reach_ids, minimum_fraction=0.01):

        if len(measurements) != len(reach_ids):
            raise Exception('Different number of measurements and associated reach IDs entered')

        self.streams = network
        self.measurements = measurements
        self.reach_ids = reach_ids
        self.minimum_frac = minimum_fraction
        self.dn = gpd.read_file(network)

        self.reach_stats = {}  # {reach: {slope: val, drain_area: val, d16: val, d50: val, d84: val}, reach: {}}
        for i, meas in enumerate(measurements):
            df = pd.read_csv(meas)
            df['D'] = df['D']/1000  # converting from mm to m

            # bootstrap the data to get range
            num_boots = 100
            boot_medians = []
            boot_d16 = []
            boot_d84 = []
            for j in range(num_boots):
                boot = resample(df['D'], replace=True, n_samples=len(0.8*df['D']))
                boot_medians.append(np.median(boot))
                boot_d16.append(np.percentile(boot, 16))
                boot_d84.append(np.percentile(boot, 84))
            medgr, d16gr, d84gr = np.asarray(boot_medians), np.asarray(boot_d16), np.asarray(boot_d84)
            mean_d, mean_d16, mean_d84 = np.mean(medgr), np.mean(d16gr), np.mean(d84gr)
            low_d50 = mean_d - (1.64 * np.std(medgr))  # does there need to be a min
            high_d50 = mean_d + (1.64 * np.std(medgr))
            low_d16 = mean_d16 - (1.64 * np.std(d16gr))
            high_d16 = mean_d16 + (1.64 * np.std(d16gr))
            low_d84 = mean_d84 - (1.64 * np.std(d84gr))
            high_d84 = mean_d84 + (1.64 * np.std(d84gr))

            self.reach_stats[reach_ids[i]] = {
                'slope': self.dn.loc[reach_ids[i], 'Slope'],
                'drain_area': self.dn.loc[reach_ids[i], 'TotDASqKm'],  # might need to change to DivDA
                'd16': np.percentile(df['D'], 16),
                'd16_low': low_d16,
                'd16_high': high_d16,
                'd50': np.median(df['D']),
                'd50_low': low_d50,
                'd50_high': high_d50,
                'd84': np.percentile(df['D'], 84),
                'd84_low': low_d84,
                'd84_high': high_d84
            }

        self.find_crit_depth()
        self.get_hydraulic_geom_coef()
        self.gs = self.estimate_grain_size()
        self.grain_size_distributions()
        self.save_network()
        self.save_grain_size_dict()

    def find_crit_depth(self):  # for now only option is using my critical shields method; could addd fixed eg 0.045

        rho_s = 2650  # density of sediment
        rho = 1000  # density of water

        for reachid, vals in self.reach_stats.items():

            tau_c_star_16 = 0.038*(vals['d16']/vals['d50'])**-0.65
            tau_c_16 = tau_c_star_16 * (rho_s-rho) * 9.81 * vals['d16']
            h_c_16 = tau_c_16 / (rho * 9.81 * vals['slope'])
            tau_c_16_low = tau_c_star_16 * (rho_s-rho) * 9.81 * vals['d16_low']
            h_c_16_low = tau_c_16_low / (rho * 9.81 * vals['slope'])
            tau_c_16_high = tau_c_star_16 * (rho_s-rho) * 9.81 * vals['d16_high']
            h_c_16_high = tau_c_16_high / (rho * 9.81 * vals['slope'])
            tau_c_star_50 = 0.038
            tau_c_50 = tau_c_star_50 * (rho_s - rho) * 9.81 * vals['d50']
            h_c_50 = tau_c_50 / (rho * 9.81 * vals['slope'])
            tau_c_50_low = tau_c_star_50 * (rho_s - rho) * 9.81 * vals['d50_low']
            h_c_50_low = tau_c_50_low / (rho * 9.81 * vals['slope'])
            tau_c_50_high = tau_c_star_50 * (rho_s - rho) * 9.81 * vals['d50_high']
            h_c_50_high = tau_c_50_high / (rho * 9.81 * vals['slope'])
            tau_c_star_84 = 0.038 * (vals['d84'] / vals['d50']) ** -0.65
            tau_c_84 = tau_c_star_84 * (rho_s - rho) * 9.81 * vals['d84']
            h_c_84 = tau_c_84 / (rho * 9.81 * vals['slope'])
            tau_c_84_low = tau_c_star_84 * (rho_s - rho) * 9.81 * vals['d84_low']
            h_c_84_low = tau_c_84_low / (rho * 9.81 * vals['slope'])
            tau_c_84_high = tau_c_star_84 * (rho_s - rho) * 9.81 * vals['d84_high']
            h_c_84_high = tau_c_84_high / (rho * 9.81 * vals['slope'])
            self.reach_stats[reachid].update({'depth_d16': h_c_16,
                                              'depth_d16_low': h_c_16_low,
                                              'depth_d16_high': h_c_16_high,
                                              'depth_d50': h_c_50,
                                              'depth_d50_low': h_c_50_low,
                                              'depth_d50_high': h_c_50_high,
                                              'depth_d84': h_c_84,
                                              'depth_d84_low': h_c_84_low,
                                              'depth_d84_high': h_c_84_high})

    # if there's only one measurement, use default hydraulic geometry
    def get_hydraulic_geom_coef(self):

        for reachid, vals in self.reach_stats.items():

            self.reach_stats[reachid].update({
                'h_coef_16': self.reach_stats[reachid]['depth_d16']/self.reach_stats[reachid]['drain_area']**0.4,
                'h_coef_16_low': self.reach_stats[reachid]['depth_d16_low']/self.reach_stats[reachid]['drain_area']**0.4,
                'h_coef_16_high': self.reach_stats[reachid]['depth_d16_high']/self.reach_stats[reachid]['drain_area']**0.4,
                'h_coef_50': self.reach_stats[reachid]['depth_d50']/self.reach_stats[reachid]['drain_area']**0.4,
                'h_coef_50_low': self.reach_stats[reachid]['depth_d50_low'] / self.reach_stats[reachid]['drain_area'] ** 0.4,
                'h_coef_50_high': self.reach_stats[reachid]['depth_d50_high'] / self.reach_stats[reachid]['drain_area'] ** 0.4,
                'h_coef_84': self.reach_stats[reachid]['depth_d84']/self.reach_stats[reachid]['drain_area']**0.4,
                'h_coef_84_low': self.reach_stats[reachid]['depth_d84_low']/self.reach_stats[reachid]['drain_area']**0.4,
                'h_coef_84_high': self.reach_stats[reachid]['depth_d84_high']/self.reach_stats[reachid]['drain_area']**0.4
            })

    # if there's more than one measurement refine hydraulic geometry relation for basin

    def estimate_grain_size(self):

        if len(self.reach_stats) == 1:

            stats_dict = {}
            for i in self.dn.index:
                print(f'Estimating D50, D16, D84 for segment : {i}')
                h_c_50 = self.reach_stats[self.reach_ids[0]]['h_coef_50']*self.dn.loc[i, 'TotDASqKm']**0.4
                tau_c_50 = 1000 * 9.81 * h_c_50 * self.dn.loc[i, 'Slope']
                d50 = tau_c_50 / ((2650 - 1000) * 9.81 * 0.038)
                self.dn.loc[i, 'D50'] = d50 * 1000

                h_c_50_low = self.reach_stats[self.reach_ids[0]]['h_coef_50_low'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
                tau_c_50_low = 1000 * 9.81 * h_c_50_low * self.dn.loc[i, 'Slope']
                d50_low = tau_c_50_low / ((2650 - 1000) * 9.81 * 0.038)
                self.dn.loc[i, 'D50_low'] = d50_low * 1000

                h_c_50_high = self.reach_stats[self.reach_ids[0]]['h_coef_50_high'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
                tau_c_50_high = 1000 * 9.81 * h_c_50_high * self.dn.loc[i, 'Slope']
                d50_high = tau_c_50_high / ((2650 - 1000) * 9.81 * 0.038)
                self.dn.loc[i, 'D50_high'] = d50_high * 1000

                h_c_16 = self.reach_stats[self.reach_ids[0]]['h_coef_16']*self.dn.loc[i, 'TotDASqKm']**0.4
                tau_c_16 = 1000 * 9.81 * h_c_16 * self.dn.loc[i, 'Slope']
                tau_c_star_16 = 0.038 * (self.reach_stats[self.reach_ids[0]]['d16']/self.reach_stats[self.reach_ids[0]]['d50']) ** -0.65
                d16 = tau_c_16 / ((2650 - 1000) * 9.81 * tau_c_star_16)
                self.dn.loc[i, 'D16'] = d16 * 1000

                h_c_16_low = self.reach_stats[self.reach_ids[0]]['h_coef_16_low']*self.dn.loc[i, 'TotDASqKm']**0.4
                tau_c_16_low = 1000 * 9.81 * h_c_16_low * self.dn.loc[i, 'Slope']
                tau_c_star_16_low = 0.038 * (self.reach_stats[self.reach_ids[0]]['d16_low']/self.reach_stats[self.reach_ids[0]]['d50']) ** -0.65
                d16_low = tau_c_16_low / ((2650 - 1000) * 9.81 * tau_c_star_16_low)
                self.dn.loc[i, 'D16_low'] = d16_low * 1000

                h_c_16_high = self.reach_stats[self.reach_ids[0]]['h_coef_16_high'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
                tau_c_16_high = 1000 * 9.81 * h_c_16_high * self.dn.loc[i, 'Slope']
                tau_c_star_16_high = 0.038 * (self.reach_stats[self.reach_ids[0]]['d16_high'] / self.reach_stats[self.reach_ids[0]]['d50']) ** -0.65
                d16_high = tau_c_16_high / ((2650 - 1000) * 9.81 * tau_c_star_16_high)
                self.dn.loc[i, 'D16_high'] = d16_high * 1000

                h_c_84 = self.reach_stats[self.reach_ids[0]]['h_coef_84'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
                tau_c_84 = 1000 * 9.81 * h_c_84 * self.dn.loc[i, 'Slope']
                tau_c_star_84 = 0.038 * (self.reach_stats[self.reach_ids[0]]['d84'] / self.reach_stats[self.reach_ids[0]]['d50']) ** -0.65
                d84 = tau_c_84 / ((2650 - 1000) * 9.81 * tau_c_star_84)
                self.dn.loc[i, 'D84'] = d84 * 1000

                h_c_84_low = self.reach_stats[self.reach_ids[0]]['h_coef_84_low'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
                tau_c_84_low = 1000 * 9.81 * h_c_84_low * self.dn.loc[i, 'Slope']
                tau_c_star_84_low = 0.038 * (self.reach_stats[self.reach_ids[0]]['d84_low'] / self.reach_stats[self.reach_ids[0]]['d50']) ** -0.65
                d84_low = tau_c_84_low / ((2650 - 1000) * 9.81 * tau_c_star_84_low)
                self.dn.loc[i, 'D84_low'] = d84_low * 1000

                h_c_84_high = self.reach_stats[self.reach_ids[0]]['h_coef_84_high'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
                tau_c_84_high = 1000 * 9.81 * h_c_84_high * self.dn.loc[i, 'Slope']
                tau_c_star_84_high = 0.038 * (self.reach_stats[self.reach_ids[0]]['d84_high'] / self.reach_stats[self.reach_ids[0]]['d50']) ** -0.65
                d84_high = tau_c_84_high / ((2650 - 1000) * 9.81 * tau_c_star_84_high)
                self.dn.loc[i, 'D84_high'] = d84_high * 1000

                stats_dict[i] = {
                    'd50': d50,
                    'd50_low': d50_low,
                    'd50_high': d50_high,
                    'd16': d16,
                    'd16_low': d16_low,
                    'd16_high': d16_high,
                    'd84': d84,
                    'd84_low': d84_low,
                    'd84_high': d84_high,
                    'fractions': {}
                }

        else:
            stats_dict = {}

        return stats_dict

    def grain_size_distributions(self):

        def med_err(loc, a, scale, med_obs):
            dist = stats.skewnorm(a, loc, scale).rvs(1000)
            dist = dist[dist > 0.0005]  # gets rid of sub-zero values (makes it sand and greater)
            med_est = np.median(dist)
            err = abs(med_est - med_obs)

            return err

        def std_err(scale, a, loc, d16, d84):
            dist = stats.skewnorm(a, loc, scale).rvs(1000)
            dist = dist[dist > 0.0005]
            err = abs(np.percentile(dist, 16) - d16)**2 + abs(np.percentile(dist, 84) - d84)**2

            return err

        a_init, loc_init, scale_init = 5, 0.015, 0.05

        if len(self.measurements) == 1:
            data = pd.read_csv(self.measurements[0])
            data = data['D']/1000

            a, loc, scale = stats.skewnorm.fit(data)

            # do i want to produce any plots..?

            for i in self.dn.index:
                print(f'finding distribution for segment {i}')
                d_50, d_16, d_84 = (self.dn.loc[i, 'D50']/1000), (self.dn.loc[i, 'D16_low']/1000), (self.dn.loc[i, 'D84_high']/1000)

                res = minimize(med_err, loc, args=(a, scale, d_50), method='Nelder-Mead', tol=0.001)
                loc_opt = res.x[0]
                if res.success is False:
                    print('median optimization issue: ', res.message)

                res2 = minimize(std_err, scale, args=(a, loc_opt, d_16, d_84), method='Nelder-Mead', tol=0.003)
                scale_opt = res2.x[0]
                if res2.success is False:
                    print('d16/d84 optimization issue: ', res2.message)

                new_data = stats.skewnorm(a, loc_opt, scale_opt).rvs(1000)

                self.gs[i]['fractions'].update({
                    '0-1': {'fraction': sum(1 for d in new_data if 0 < d <= 0.001) / len(new_data),
                            'lower': 0,
                            'upper': 0.001},
                    '1-2': {'fraction': sum(1 for d in new_data if 0.001 < d <= 0.002) / len(new_data),
                            'lower': 0.001,
                            'upper': 0.002},
                    '2-4': {'fraction': sum(1 for d in new_data if 0.002 < d <= 0.004) / len(new_data),
                            'lower': 0.002,
                            'upper': 0.004},
                    '4-6': {'fraction': sum(1 for d in new_data if 0.004 < d <= 0.006) / len(new_data),
                            'lower': 0.004,
                            'upper': 0.006},
                    '6-8': {'fraction': sum(1 for d in new_data if 0.006 < d <= 0.008) / len(new_data),
                            'lower': 0.006,
                            'upper': 0.008},
                    '8-12': {'fraction': sum(1 for d in new_data if 0.008 < d <= 0.012) / len(new_data),
                             'lower': 0.008,
                             'upper': 0.012},
                    '12-16': {'fraction': sum(1 for d in new_data if 0.012 < d <= 0.016) / len(new_data),
                              'lower': 0.012,
                              'upper': 0.016},
                    '16-24': {'fraction': sum(1 for d in new_data if 0.016 < d <= 0.024) / len(new_data),
                              'lower': 0.016,
                              'upper': 0.024},
                    '24-32': {'fraction': sum(1 for d in new_data if 0.024 < d <= 0.032) / len(new_data),
                              'lower': 0.024,
                              'upper': 0.032},
                    '32-48': {'fraction': sum(1 for d in new_data if 0.032 < d <= 0.048) / len(new_data),
                              'lower': 0.032,
                              'upper': 0.048},
                    '48-64': {'fraction': sum(1 for d in new_data if 0.048 < d <= 0.064) / len(new_data),
                              'lower': 0.048,
                              'upper': 0.064},
                    '64-96': {'fraction': sum(1 for d in new_data if 0.064 < d <= 0.096) / len(new_data),
                              'lower': 0.064,
                              'upper': 0.096},
                    '96-128': {'fraction': sum(1 for d in new_data if 0.096 < d <= 0.128) / len(new_data),
                               'lower': 0.096,
                               'upper': 0.128},
                    '128-192': {'fraction': sum(1 for d in new_data if 0.128 < d <= 0.192) / len(new_data),
                                'lower': 0.128,
                                'upper': 0.192},
                    '192-256': {'fraction': sum(1 for d in new_data if 0.192 < d <= 0.256) / len(new_data),
                                'lower': 0.192,
                                'upper': 256},
                    '>256': {'fraction': sum(1 for d in new_data if 0.256 < d) / len(new_data),
                             'lower': 0.256,
                             'upper': 512}
                })

                if i in [11, 25]:
                    plt.figure()
                    plt.hist(new_data, bins=15, density=True, alpha=0.6, color='g')
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = stats.skewnorm.pdf(x, a, loc_opt, scale_opt)
                    plt.plot(x, p, 'k', linewidth=2)
                    plt.show()

    def save_network(self):
        self.dn.to_file(self.streams)

    def save_grain_size_dict(self):
        with open(os.path.join(os.path.dirname(self.streams), 'stream_segment_grain_sizes.json'), 'w') as f:
            json.dump(self.gs, f, indent=4)


if __name__ == '__main__':

    GrainSize(network='./Input_data/NHDPlus_Woods.shp', measurements=['./Input_data/Woods_D.csv'], reach_ids=[41])
