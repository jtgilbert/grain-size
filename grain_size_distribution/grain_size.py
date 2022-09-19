import os
import json
import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.utils import resample
from scipy import stats
from scipy.optimize import fmin


class GrainSize:
    """
    Estimates D50, D16, and D84 as well as the 95% confidence interval for each, and a full grain size distribution
    (fraction of the bed surface in each half-phi size bin) for every segment of
    a drainage network using one or more field measurements of grain size distribution
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
            df['D'] = df['D'] / 1000  # converting from mm to m

            # bootstrap the data to get range
            num_boots = 100
            boot_medians = []
            boot_d16 = []
            boot_d84 = []
            for j in range(num_boots):
                boot = resample(df['D'], replace=True, n_samples=len(0.8 * df['D']))
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
            tau_c_star_16 = 0.038 * (vals['d16'] / vals['d50']) ** -0.65
            tau_c_16 = tau_c_star_16 * (rho_s - rho) * 9.81 * vals['d16']
            h_c_16 = tau_c_16 / (rho * 9.81 * vals['slope'])
            tau_c_16_low = tau_c_star_16 * (rho_s - rho) * 9.81 * vals['d16_low']
            h_c_16_low = tau_c_16_low / (rho * 9.81 * vals['slope'])
            tau_c_16_high = tau_c_star_16 * (rho_s - rho) * 9.81 * vals['d16_high']
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

    def get_hydraulic_geom_coef(self):

        for reachid, vals in self.reach_stats.items():
            self.reach_stats[reachid].update({
                'h_coef_16': self.reach_stats[reachid]['depth_d16'] / self.reach_stats[reachid]['drain_area'] ** 0.4,
                'h_coef_16_low': self.reach_stats[reachid]['depth_d16_low'] / self.reach_stats[reachid][
                    'drain_area'] ** 0.4,
                'h_coef_16_high': self.reach_stats[reachid]['depth_d16_high'] / self.reach_stats[reachid][
                    'drain_area'] ** 0.4,
                'h_coef_50': self.reach_stats[reachid]['depth_d50'] / self.reach_stats[reachid]['drain_area'] ** 0.4,
                'h_coef_50_low': self.reach_stats[reachid]['depth_d50_low'] / self.reach_stats[reachid][
                    'drain_area'] ** 0.4,
                'h_coef_50_high': self.reach_stats[reachid]['depth_d50_high'] / self.reach_stats[reachid][
                    'drain_area'] ** 0.4,
                'h_coef_84': self.reach_stats[reachid]['depth_d84'] / self.reach_stats[reachid]['drain_area'] ** 0.4,
                'h_coef_84_low': self.reach_stats[reachid]['depth_d84_low'] / self.reach_stats[reachid][
                    'drain_area'] ** 0.4,
                'h_coef_84_high': self.reach_stats[reachid]['depth_d84_high'] / self.reach_stats[reachid][
                    'drain_area'] ** 0.4
            })

        # if there's only one measurement, use the calculated coefficient
        if len(self.reach_ids) == 1:
            self.reach_stats.update({'coefs':
                                         {'h_coef_16': self.reach_stats[self.reach_ids[0]]['h_coef_16'],
                                          'h_coef_16_low': self.reach_stats[self.reach_ids[0]]['h_coef_16_low'],
                                          'h_coef_16_high': self.reach_stats[self.reach_ids[0]]['h_coef_16_high'],
                                          'h_coef_50': self.reach_stats[self.reach_ids[0]]['h_coef_50'],
                                          'h_coef_50_low': self.reach_stats[self.reach_ids[0]]['h_coef_50_low'],
                                          'h_coef_50_high': self.reach_stats[self.reach_ids[0]]['h_coef_50_high'],
                                          'h_coef_84': self.reach_stats[self.reach_ids[0]]['h_coef_84'],
                                          'h_coef_84_low': self.reach_stats[self.reach_ids[0]]['h_coef_84_low'],
                                          'h_coef_84_high': self.reach_stats[self.reach_ids[0]]['h_coef_84_high']},
                                     'coef_type': 'value'
                                     })

        # if there's more than one measurement, find relationship between coefficient and reach slope
        elif len(self.reach_ids) > 1:
            coefs = {'16': {'low': [], 'mid': [], 'high': [], 'slope': []},
                     '50': {'low': [], 'mid': [], 'high': [], 'slope': []},
                     '84': {'low': [], 'mid': [], 'high': [], 'slope': []},
                     }

            for reachid, vals in self.reach_stats.items():
                coefs['16']['low'].append(vals['h_coef_16_low'])
                coefs['16']['mid'].append(vals['h_coef_16'])
                coefs['16']['high'].append(vals['h_coef_16_high'])
                coefs['16']['slope'].append(vals['slope'])
                coefs['50']['low'].append(vals['h_coef_50_low'])
                coefs['50']['mid'].append(vals['h_coef_50'])
                coefs['50']['high'].append(vals['h_coef_50_high'])
                coefs['50']['slope'].append(vals['slope'])
                coefs['84']['low'].append(vals['h_coef_84_low'])
                coefs['84']['mid'].append(vals['h_coef_84'])
                coefs['84']['high'].append(vals['h_coef_84_high'])
                coefs['84']['slope'].append(vals['slope'])

            low_16_a, low_16_b = np.polyfit(coefs['16']['slope'], coefs['16']['low'], 1)
            mid_16_a, mid_16_b = np.polyfit(coefs['16']['slope'], coefs['16']['mid'], 1)
            high_16_a, high_16_b = np.polyfit(coefs['16']['slope'], coefs['16']['high'], 1)
            low_50_a, low_50_b = np.polyfit(coefs['50']['slope'], coefs['50']['low'], 1)
            mid_50_a, mid_50_b = np.polyfit(coefs['50']['slope'], coefs['50']['mid'], 1)
            high_50_a, high_50_b = np.polyfit(coefs['50']['slope'], coefs['50']['high'], 1)
            low_84_a, low_84_b = np.polyfit(coefs['84']['slope'], coefs['84']['low'], 1)
            mid_84_a, mid_84_b = np.polyfit(coefs['84']['slope'], coefs['84']['mid'], 1)
            high_84_a, high_84_b = np.polyfit(coefs['84']['slope'], coefs['84']['high'], 1)

            self.reach_stats.update({'coefs':
                                         {'h_coef_16': [mid_16_a, mid_16_b],
                                          'h_coef_16_low': [low_16_a, low_16_b],
                                          'h_coef_16_high': [high_16_a, high_16_b],
                                          'h_coef_50': [mid_50_a, mid_50_b],
                                          'h_coef_50_low': [low_50_a, low_50_b],
                                          'h_coef_50_high': [high_50_a, high_50_b],
                                          'h_coef_84': [mid_84_a, mid_84_b],
                                          'h_coef_84_low': [low_84_a, low_84_b],
                                          'h_coef_84_high': [high_84_a, high_84_b]},
                                     'coef_type': 'function'
                                     })

    def estimate_grain_size(self):

        stats_dict = {}
        for i in self.dn.index:
            print(f'Estimating D50, D16, D84 for segment : {i}')
            if self.reach_stats['coef_type'] == 'value':
                h_c_50 = self.reach_stats['coefs']['h_coef_50'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_50'][0]*self.dn.loc[i, 'Slope'] + self.reach_stats['coefs']['h_coef_50'][1]
                h_c_50 = coefval * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            tau_c_50 = 1000 * 9.81 * h_c_50 * self.dn.loc[i, 'Slope']
            d50 = tau_c_50 / ((2650 - 1000) * 9.81 * 0.038)
            self.dn.loc[i, 'D50'] = d50 * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_50_low = self.reach_stats['coefs']['h_coef_50_low'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_50_low'][0] * self.dn.loc[i, 'Slope'] + self.reach_stats['coefs']['h_coef_50_low'][1]
                h_c_50_low = coefval * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            tau_c_50_low = 1000 * 9.81 * h_c_50_low * self.dn.loc[i, 'Slope']
            d50_low = tau_c_50_low / ((2650 - 1000) * 9.81 * 0.038)
            self.dn.loc[i, 'D50_low'] = d50_low * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_50_high = self.reach_stats['coefs']['h_coef_50_high'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_50_high'][0] * self.dn.loc[i, 'Slope'] + self.reach_stats['coefs']['h_coef_50_high'][1]
                h_c_50_high = coefval * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            tau_c_50_high = 1000 * 9.81 * h_c_50_high * self.dn.loc[i, 'Slope']
            d50_high = tau_c_50_high / ((2650 - 1000) * 9.81 * 0.038)
            self.dn.loc[i, 'D50_high'] = d50_high * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_16 = self.reach_stats['coefs']['h_coef_16'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_16'][0] * self.dn.loc[i, 'Slope'] + self.reach_stats['coefs']['h_coef_16'][1]
                h_c_16 = coefval * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            tau_c_16 = 1000 * 9.81 * h_c_16 * self.dn.loc[i, 'Slope']
            tau_c_star_16 = 0.038 * (
                        self.reach_stats[self.reach_ids[0]]['d16'] / self.reach_stats[self.reach_ids[0]][
                    'd50']) ** -0.65
            d16 = tau_c_16 / ((2650 - 1000) * 9.81 * tau_c_star_16)
            self.dn.loc[i, 'D16'] = d16 * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_16_low = self.reach_stats['coefs']['h_coef_16_low'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_16_low'][0] * self.dn.loc[i, 'Slope'] + self.reach_stats['coefs']['h_coef_16_low'][1]
                h_c_16_low = coefval * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            tau_c_16_low = 1000 * 9.81 * h_c_16_low * self.dn.loc[i, 'Slope']
            tau_c_star_16_low = 0.038 * (
                        self.reach_stats[self.reach_ids[0]]['d16_low'] / self.reach_stats[self.reach_ids[0]][
                    'd50']) ** -0.65
            d16_low = tau_c_16_low / ((2650 - 1000) * 9.81 * tau_c_star_16_low)
            self.dn.loc[i, 'D16_low'] = d16_low * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_16_high = self.reach_stats['coefs']['h_coef_16_high'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_16_high'][0] * self.dn.loc[i, 'Slope'] + self.reach_stats['coefs']['h_coef_16_high'][1]
                h_c_16_high = coefval * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            tau_c_16_high = 1000 * 9.81 * h_c_16_high * self.dn.loc[i, 'Slope']
            tau_c_star_16_high = 0.038 * (
                        self.reach_stats[self.reach_ids[0]]['d16_high'] / self.reach_stats[self.reach_ids[0]][
                    'd50']) ** -0.65
            d16_high = tau_c_16_high / ((2650 - 1000) * 9.81 * tau_c_star_16_high)
            self.dn.loc[i, 'D16_high'] = d16_high * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_84 = self.reach_stats['coefs']['h_coef_84'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_84'][0] * self.dn.loc[i, 'Slope'] + self.reach_stats['coefs']['h_coef_84'][1]
                h_c_84 = coefval * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            tau_c_84 = 1000 * 9.81 * h_c_84 * self.dn.loc[i, 'Slope']
            tau_c_star_84 = 0.038 * (
                        self.reach_stats[self.reach_ids[0]]['d84'] / self.reach_stats[self.reach_ids[0]][
                    'd50']) ** -0.65
            d84 = tau_c_84 / ((2650 - 1000) * 9.81 * tau_c_star_84)
            self.dn.loc[i, 'D84'] = d84 * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_84_low = self.reach_stats['coefs']['h_coef_84_low'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_84_low'][0] * self.dn.loc[i, 'Slope'] + self.reach_stats['coefs']['h_coef_84_low'][1]
                h_c_84_low = coefval * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            tau_c_84_low = 1000 * 9.81 * h_c_84_low * self.dn.loc[i, 'Slope']
            tau_c_star_84_low = 0.038 * (
                        self.reach_stats[self.reach_ids[0]]['d84_low'] / self.reach_stats[self.reach_ids[0]][
                    'd50']) ** -0.65
            d84_low = tau_c_84_low / ((2650 - 1000) * 9.81 * tau_c_star_84_low)
            self.dn.loc[i, 'D84_low'] = d84_low * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_84_high = self.reach_stats['coefs']['h_coef_84_high'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_84_high'][0] * self.dn.loc[i, 'Slope'] + self.reach_stats['coefs']['h_coef_84_high'][1]
                h_c_84_high = coefval * self.dn.loc[i, 'TotDASqKm'] ** 0.4
            tau_c_84_high = 1000 * 9.81 * h_c_84_high * self.dn.loc[i, 'Slope']
            tau_c_star_84_high = 0.038 * (
                        self.reach_stats[self.reach_ids[0]]['d84_high'] / self.reach_stats[self.reach_ids[0]][
                    'd50']) ** -0.65
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

        return stats_dict

    def grain_size_distributions(self):

        def med_err(loc, scale, med_obs):
            dist = stats.norm(loc, scale).rvs(10000)
            dist = dist**2
            dist = dist[dist > 0.0005]  # gets rid of sub-zero values (makes it sand and greater)
            med_est = np.percentile(dist, 50)
            err = (med_obs - med_est) ** 2

            return err

        def std_err(scale, loc, d16, d84):
            dist = stats.norm(loc, scale).rvs(10000)
            dist = dist**2
            dist = dist[dist > 0.0005]
            err = (np.percentile(dist, 16) - d16) ** 2 + (np.percentile(dist, 84) - d84) ** 2

            return err

        if len(self.measurements) == 1:
            data = pd.read_csv(self.measurements[0])
            data = np.sqrt(data['D'] / 1000)

            k2, pval = stats.normaltest(data)
            if pval < 1e-3:
                print('p-value: ', pval, ' indicates that the distribution may not be normal')

            params = stats.norm.fit(data)
            loc = params[0]
            scale = params[1]

        else:  # for now just take average...could maybe make function of slope too
            loclist, scalelist = [], []
            for i, meas in enumerate(self.measurements):
                data = pd.read_csv(meas)
                data = np.sqrt(data['D'] / 1000)

                k2, pval = stats.normaltest(data)
                if pval < 1e-3:
                    print('p-value: ', pval, ' indicates that the distribution may not be normal')

                params = stats.norm.fit(data)
                loclist.append(params[0])
                scalelist.append(params[1])
            loc, scale = np.average(loclist), np.average(scalelist)

            # do i want to produce any plots..?

        for i in self.dn.index:
            print(f'finding distribution for segment {i}')
            d_50, d_16, d_84 = (self.dn.loc[i, 'D50'] / 1000), (self.dn.loc[i, 'D16_low'] / 1000), (
                        self.dn.loc[i, 'D84_high'] / 1000)

            res = fmin(med_err, loc, args=(scale, d_50))
            loc_opt = res[0]

            res2 = fmin(std_err, scale, args=(loc_opt, d_16, d_84))
            scale_opt = res2[0]
            # while scale_opt <= 0:
            #    res2 = fmin(std_err, np.array((abs(scale_opt), a)), args=(loc_opt, d_84-d_16))
            #    scale_opt = res2[0]

            new_data = stats.norm(loc_opt, scale_opt).rvs(10000)
            new_data = new_data**2
            new_data = new_data[new_data >= 0.0005]

            self.gs[i].update({'Dmax': max(new_data)})
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
                '4-8': {'fraction': sum(1 for d in new_data if 0.004 < d <= 0.008) / len(new_data),
                        'lower': 0.004,
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

            # if i in [11, 25]:
            #     plt.figure()
            #     plt.hist(new_data, bins=15, density=True, alpha=0.6, color='g')
            #     xmin, xmax = plt.xlim()
            #     x = np.linspace(xmin, xmax, 100)
            #     p = stats.skewnorm.pdf(x, a, loc_opt, scale_opt)
            #     plt.plot(x, p, 'k', linewidth=2)
            #     plt.show()

    def save_network(self):
        self.dn.to_file(self.streams)

    def save_grain_size_dict(self):
        with open(os.path.join(os.path.dirname(self.streams), 'stream_segment_grain_sizes.json'), 'w') as f:
            json.dump(self.gs, f, indent=4)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('stream_network', help='Path to stream network feature class', type=str)
    parser.add_argument('--measurements', help='A list of paths to csv files containing grain size measurements; '
                                             'should have header "D" at top of column followed by individual '
                                             'grain size measurements',
                        nargs='+', type=str, required=True)
    parser.add_argument('--reach_ids', help='A list containing the reach IDs from the stream network feature class'
                                          'associated with the grain size measurements, in the same order as the '
                                          'measurements',
                        nargs='+', type=int, required=True)
    args = parser.parse_args()

    GrainSize(args.stream_network, args.measurements, args.reach_ids)


if __name__ == '__main__':
    main()
    #GrainSize(network='./Input_data/NHDPlus_Woods.shp', measurements=['./Input_data/Woods_D.csv'], reach_ids=[41])
