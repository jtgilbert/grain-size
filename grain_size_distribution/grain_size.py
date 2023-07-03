import os
import json
import argparse
import geopandas as gpd
import numpy as np
import pandas as pd

from typing import List
from sklearn.utils import resample
from scipy import stats
from scipy.optimize import fmin


class GrainSize:
    """
    Estimates D50, D16, and D84 as well as the 95% confidence interval for each, and a full grain size distribution
    (fraction of the bed surface in each half-phi size bin) for every segment of
    a drainage network using one or more field measurements of grain size distribution
    """

    def __init__(self, network: str, measurements, reach_ids, da_field: str,
                 max_size: float = 3.0, minimum_fraction: float = None):
        """

        :param network: path to stream network feature class
        :param measurements: path to csv files containing grain size measurements
        :param reach_ids: list of network reach ids associated with the measurements
        :param da_field: the name of the drainage area field in the network feature class
        :param max_size: a maximum size for sediment particles in the basin (default = 3 m)
        :param minimum_fraction: a minimum fraction to assign to each size class of a grain size distribution
        (default = 0.005, 0.5%)
        """

        if len(measurements) != len(reach_ids):
            raise Exception('Different number of measurements and associated reach IDs entered')

        self.streams = network
        self.measurements = measurements
        self.reach_ids = reach_ids
        self.da_field = da_field
        self.max_size = max_size
        self.minimum_frac = minimum_fraction
        self.dn = gpd.read_file(network)

        # check for drainage area field and slope field in input network
        if da_field not in self.dn.columns:
            raise Exception(f'Drainage Area field {da_field} missing from input drainage network attributes')
        if 'Slope' not in self.dn.columns:
            raise Exception("'Slope' field missing from input drainage network attributes")

        self.reach_stats = {}  # {reach: {slope: val, drain_area: val, d16: val, d50: val, d84: val}, reach: {}}
        for i, meas in enumerate(measurements):
            df = pd.read_csv(meas)
            df['D'] = df['D'] / 1000  # converting from mm to m

            dmax = max(df['D'])

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
                'drain_area': self.dn.loc[reach_ids[i], self.da_field],
                'd16': np.percentile(df['D'], 16),
                'd16_low': low_d16,
                'd16_high': high_d16,
                'd50': np.median(df['D']),
                'd50_low': low_d50,
                'd50_high': high_d50,
                'd84': np.percentile(df['D'], 84),
                'd84_low': low_d84,
                'd84_high': high_d84,
                'dmax': dmax
            }

        # generate function for predicting roughness, and d16/d50 from slope if there's more than one measurement
        if len(self.reach_stats) > 1:
            roughness = [stats['d84'] / stats['d50'] for id, stats in self.reach_stats.items()]
            d16_rat = [stats['d16'] / stats['d50'] for id, stats in self.reach_stats.items()]
            slope = [stats['slope'] for id, stats in self.reach_stats.items()]
            rough_params = np.polyfit(slope, roughness, 1)
            d16_rat_params = np.polyfit(slope, d16_rat, 1)
            # might need to check that it can't be <0
            self.rough_coef, self.rough_rem = rough_params[0], rough_params[1]
            self.d16_coef, self.d16_rem = d16_rat_params[0], d16_rat_params[1]
            self.roughness_type = 'function'

        else:
            self.roughness = self.reach_stats[self.reach_ids[0]]['d84']/self.reach_stats[self.reach_ids[0]]['d50']
            self.d16_rat = self.reach_stats[self.reach_ids[0]]['d16']/self.reach_stats[self.reach_ids[0]]['d50']
            self.roughness_type = 'value'

        # get dmax params or scalar
        if len(self.reach_stats) > 1:
            maxvals = [atts['dmax'] for _, atts in self.reach_stats.items()]
            maxd84s = [atts['d84'] for _, atts in self.reach_stats.items()]
            slopevals = [atts['slope'] for _, atts in self.reach_stats.items()]
            self.dmax_params = np.polyfit(slopevals, maxvals, 1)
            self.dmax_ratio = np.average([maxvals[i]/maxd84s[i] for i in range(len(maxvals))])
        else:
            self.dmax_params = [atts['dmax']/atts['d84'] for _, atts in self.reach_stats.items()]
            self.dmax_ratio = self.dmax_params[0]

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
            roughness = vals['d84'] / vals['d50']
            if roughness <= 2:
                tau_coef = 0.025
            elif 2 < roughness < 3.5:
                tau_coef = 0.087 * np.log(roughness) - 0.034
            else:
                tau_coef = 0.073
            tau_c_star_16 = tau_coef * (vals['d16'] / vals['d50']) ** -0.68
            tau_c_16 = tau_c_star_16 * (rho_s - rho) * 9.81 * vals['d16']
            h_c_16 = tau_c_16 / (rho * 9.81 * vals['slope'])
            tau_c_16_low = tau_c_star_16 * (rho_s - rho) * 9.81 * vals['d16_low']
            h_c_16_low = tau_c_16_low / (rho * 9.81 * vals['slope'])
            tau_c_16_high = tau_c_star_16 * (rho_s - rho) * 9.81 * vals['d16_high']
            h_c_16_high = tau_c_16_high / (rho * 9.81 * vals['slope'])
            tau_c_star_50 = tau_coef
            tau_c_50 = tau_c_star_50 * (rho_s - rho) * 9.81 * vals['d50']
            h_c_50 = tau_c_50 / (rho * 9.81 * vals['slope'])
            tau_c_50_low = tau_c_star_50 * (rho_s - rho) * 9.81 * vals['d50_low']
            h_c_50_low = tau_c_50_low / (rho * 9.81 * vals['slope'])
            tau_c_50_high = tau_c_star_50 * (rho_s - rho) * 9.81 * vals['d50_high']
            h_c_50_high = tau_c_50_high / (rho * 9.81 * vals['slope'])
            tau_c_star_84 = tau_coef * (vals['d84'] / vals['d50']) ** -0.68
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

            min_slope = self.dn['Slope'].min()
            coefs_max = 0
            for i, vals in coefs.items():
                for key in vals.keys():
                    if key != 'slope':
                        if max(vals[key]) > coefs_max:
                            coefs_max = max(vals[key])

            for i, vals in coefs.items():
                for key in vals.keys():
                    if key == 'slope':
                        vals[key].append(min_slope)
                    else:
                        vals[key].append(coefs_max + 0.01)

            logvals = np.log(coefs['16']['slope'])

            low_16_a, low_16_b = np.polyfit(np.log(coefs['16']['slope']), np.log(coefs['16']['low']), 1)
            mid_16_a, mid_16_b = np.polyfit(np.log(coefs['16']['slope']), np.log(coefs['16']['mid']), 1)
            high_16_a, high_16_b = np.polyfit(np.log(coefs['16']['slope']), np.log(coefs['16']['high']), 1)
            low_50_a, low_50_b = np.polyfit(np.log(coefs['50']['slope']), np.log(coefs['50']['low']), 1)
            mid_50_a, mid_50_b = np.polyfit(np.log(coefs['50']['slope']), np.log(coefs['50']['mid']), 1)
            high_50_a, high_50_b = np.polyfit(np.log(coefs['50']['slope']), np.log(coefs['50']['high']), 1)
            low_84_a, low_84_b = np.polyfit(np.log(coefs['84']['slope']), np.log(coefs['84']['low']), 1)
            mid_84_a, mid_84_b = np.polyfit(np.log(coefs['84']['slope']), np.log(coefs['84']['mid']), 1)
            high_84_a, high_84_b = np.polyfit(np.log(coefs['84']['slope']), np.log(coefs['84']['high']), 1)

            low_16_b, mid_16_b, high_16_b, low_50_b, mid_50_b, high_50_b, low_84_b, mid_84_b, high_84_b = np.exp(low_16_b), np.exp(mid_16_b), np.exp(high_16_b), np.exp(low_50_b), np.exp(mid_50_b), np.exp(high_50_b), np.exp(low_84_b), np.exp(mid_84_b), np.exp(high_84_b)

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
            if self.roughness_type == 'function':
                rough = self.rough_coef*self.dn.loc[i, 'Slope']+self.rough_rem
                d16_rat = self.d16_coef*self.dn.loc[i, 'Slope']+self.d16_rem
            else:
                rough = self.roughness
                d16_rat = self.d16_rat
            if rough <= 2:
                tau_coef = 0.029
            else:
                tau_coef = 0.043 * np.log(rough) - 0.0005

            if d16_rat < 0.1:
                d16_rat = 0.1  # this is kind of janky to just set a fixed value...

            if self.reach_stats['coef_type'] == 'value':
                h_c_50 = self.reach_stats['coefs']['h_coef_50'] * self.dn.loc[i, self.da_field] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_50'][1]*self.dn.loc[i, 'Slope']**self.reach_stats['coefs']['h_coef_50'][0]
                h_c_50 = coefval * self.dn.loc[i, self.da_field] ** 0.4
            tau_c_50 = 1000 * 9.81 * h_c_50 * self.dn.loc[i, 'Slope']
            d50 = tau_c_50 / ((2650 - 1000) * 9.81 * tau_coef)
            self.dn.loc[i, 'D50'] = d50 * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_50_low = self.reach_stats['coefs']['h_coef_50_low'] * self.dn.loc[i, self.da_field] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_50_low'][1] * self.dn.loc[i, 'Slope']**self.reach_stats['coefs']['h_coef_50_low'][0]
                h_c_50_low = coefval * self.dn.loc[i, self.da_field] ** 0.4
            tau_c_50_low = 1000 * 9.81 * h_c_50_low * self.dn.loc[i, 'Slope']
            d50_low = tau_c_50_low / ((2650 - 1000) * 9.81 * tau_coef)
            self.dn.loc[i, 'D50_low'] = d50_low * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_50_high = self.reach_stats['coefs']['h_coef_50_high'] * self.dn.loc[i, self.da_field] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_50_high'][1] * self.dn.loc[i, 'Slope']**self.reach_stats['coefs']['h_coef_50_high'][0]
                h_c_50_high = coefval * self.dn.loc[i, self.da_field] ** 0.4
            tau_c_50_high = 1000 * 9.81 * h_c_50_high * self.dn.loc[i, 'Slope']
            d50_high = tau_c_50_high / ((2650 - 1000) * 9.81 * tau_coef)
            self.dn.loc[i, 'D50_high'] = d50_high * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_16 = self.reach_stats['coefs']['h_coef_16'] * self.dn.loc[i, self.da_field] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_16'][1] * self.dn.loc[i, 'Slope']**self.reach_stats['coefs']['h_coef_16'][0]
                h_c_16 = coefval * self.dn.loc[i, self.da_field] ** 0.4
            tau_c_16 = 1000 * 9.81 * h_c_16 * self.dn.loc[i, 'Slope']
            tau_c_star_16 = tau_coef * d16_rat ** -0.68
            d16 = tau_c_16 / ((2650 - 1000) * 9.81 * tau_c_star_16)
            self.dn.loc[i, 'D16'] = d16 * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_16_low = self.reach_stats['coefs']['h_coef_16_low'] * self.dn.loc[i, self.da_field] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_16_low'][1] * self.dn.loc[i, 'Slope']**self.reach_stats['coefs']['h_coef_16_low'][0]
                h_c_16_low = coefval * self.dn.loc[i, self.da_field] ** 0.4
            tau_c_16_low = 1000 * 9.81 * h_c_16_low * self.dn.loc[i, 'Slope']
            tau_c_star_16_low = tau_coef * d16_rat ** -0.68
            d16_low = tau_c_16_low / ((2650 - 1000) * 9.81 * tau_c_star_16_low)
            self.dn.loc[i, 'D16_low'] = d16_low * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_16_high = self.reach_stats['coefs']['h_coef_16_high'] * self.dn.loc[i, self.da_field] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_16_high'][1] * self.dn.loc[i, 'Slope']**self.reach_stats['coefs']['h_coef_16_high'][0]
                h_c_16_high = coefval * self.dn.loc[i, self.da_field] ** 0.4
            tau_c_16_high = 1000 * 9.81 * h_c_16_high * self.dn.loc[i, 'Slope']
            tau_c_star_16_high = tau_coef * d16_rat ** -0.68
            d16_high = tau_c_16_high / ((2650 - 1000) * 9.81 * tau_c_star_16_high)
            self.dn.loc[i, 'D16_high'] = d16_high * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_84 = self.reach_stats['coefs']['h_coef_84'] * self.dn.loc[i, self.da_field] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_84'][1] * self.dn.loc[i, 'Slope']**self.reach_stats['coefs']['h_coef_84'][0]
                h_c_84 = coefval * self.dn.loc[i, self.da_field] ** 0.4
            tau_c_84 = 1000 * 9.81 * h_c_84 * self.dn.loc[i, 'Slope']
            tau_c_star_84 = tau_coef * rough ** -0.68
            d84 = tau_c_84 / ((2650 - 1000) * 9.81 * tau_c_star_84)
            self.dn.loc[i, 'D84'] = d84 * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_84_low = self.reach_stats['coefs']['h_coef_84_low'] * self.dn.loc[i, self.da_field] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_84_low'][1] * self.dn.loc[i, 'Slope']**self.reach_stats['coefs']['h_coef_84_low'][0]
                h_c_84_low = coefval * self.dn.loc[i, self.da_field] ** 0.4
            tau_c_84_low = 1000 * 9.81 * h_c_84_low * self.dn.loc[i, 'Slope']
            tau_c_star_84_low = tau_coef * rough ** -0.68
            d84_low = tau_c_84_low / ((2650 - 1000) * 9.81 * tau_c_star_84_low)
            self.dn.loc[i, 'D84_low'] = d84_low * 1000

            if self.reach_stats['coef_type'] == 'value':
                h_c_84_high = self.reach_stats['coefs']['h_coef_84_high'] * self.dn.loc[i, self.da_field] ** 0.4
            else:
                coefval = self.reach_stats['coefs']['h_coef_84_high'][1] * self.dn.loc[i, 'Slope']**self.reach_stats['coefs']['h_coef_84_high'][0]
                h_c_84_high = coefval * self.dn.loc[i, self.da_field] ** 0.4
            tau_c_84_high = 1000 * 9.81 * h_c_84_high * self.dn.loc[i, 'Slope']
            tau_c_star_84_high = tau_coef * rough ** -0.68
            d84_high = tau_c_84_high / ((2650 - 1000) * 9.81 * tau_c_star_84_high)
            self.dn.loc[i, 'D84_high'] = d84_high * 1000

            if len(self.dmax_params) > 1:
                dmax = self.dn.loc[i, 'Slope']*self.dmax_params[0] + self.dmax_params[1]
            else:
                dmax = d84 * self.dmax_ratio
            if dmax < d84:
                dmax = d84 * self.dmax_ratio
            if dmax > self.max_size:
                dmax = self.max_size

            self.dn.loc[i, 'Dmax'] = dmax * 1000

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
                'dmax': dmax,
                'fractions': {}
            }

        return stats_dict

    def grain_size_distributions(self):

        def loc_err(loc, a, scale, d50):
            dist = stats.skewnorm(a, loc, scale).rvs(10000)
            d50_est = np.percentile(dist, 50)
            err = (np.log2(d50) - d50_est) ** 2

            return err

        def scale_err(scale, a, loc, d16, d84):
            if scale <= 0:
                scale = abs(scale)
            dist = stats.skewnorm(a, loc, scale).rvs(10000)
            err = (np.percentile(dist, 16) - np.log2(d16)) ** 2 + (np.percentile(dist, 84) - np.log2(d84)) ** 2

            return err

        if len(self.measurements) == 1:
            data = pd.read_csv(self.measurements[0])
            data = [np.log2(i) for i in list(data['D'])]

            params = stats.skewnorm.fit(data)
            a, loc, scale = params[0], params[1], params[2]

        else:  # for now just take average...could maybe make function of slope too
            alist, loclist, scalelist = [], [], []
            for i, meas in enumerate(self.measurements):
                data = pd.read_csv(meas)
                data = [np.log2(i) for i in list(data['D'])]

                params = stats.skewnorm.fit(data)
                alist.append(params[0])
                loclist.append(params[1])
                scalelist.append(params[2])
            a, loc, scale = np.average(alist), np.average(loclist), np.average(scalelist)

        for i in self.dn.index:
            print(f'finding distribution for segment {i}')
            d_50, d_16, d_84 = self.dn.loc[i, 'D50'], self.dn.loc[i, 'D16'], self.dn.loc[i, 'D84']

            res = fmin(loc_err, loc, args=(a, scale, d_50))
            loc_opt = res[0]

            res2 = fmin(scale_err, scale, args=(a, loc_opt, d_16, d_84))
            scale_opt = res2[0]
            # while scale_opt <= 0:
            #    res2 = fmin(scale_err, abs(scale_opt), args=(a, loc_opt, d_50))
            #    scale_opt = res2[0]

            new_data = stats.skewnorm(a, loc_opt, scale_opt).rvs(10000)
            new_data = new_data[new_data>=-1]

            self.gs[i]['fractions'].update({
                '1': {'fraction': sum(1 for d in new_data if -1 <= d < 0) / len(new_data),
                        'lower': 0.0005,
                        'upper': 0.001},
                '0': {'fraction': sum(1 for d in new_data if 0 <= d < 1) / len(new_data),
                        'lower': 0.001,
                        'upper': 0.002},
                '-1': {'fraction': sum(1 for d in new_data if 1 <= d < 2) / len(new_data),
                        'lower': 0.002,
                        'upper': 0.004},
                '-2': {'fraction': sum(1 for d in new_data if 2 <= d < 2.5) / len(new_data),
                        'lower': 0.004,
                        'upper': 0.0057},
                '-2.5': {'fraction': sum(1 for d in new_data if 2.5 <= d < 3) / len(new_data),
                         'lower': 0.0057,
                         'upper': 0.008},
                '-3': {'fraction': sum(1 for d in new_data if 3 <= d < 3.5) / len(new_data),
                          'lower': 0.008,
                          'upper': 0.0113},
                '-3.5': {'fraction': sum(1 for d in new_data if 3.5 <= d < 4) / len(new_data),
                          'lower': 0.0113,
                          'upper': 0.016},
                '-4': {'fraction': sum(1 for d in new_data if 4 <= d < 4.5) / len(new_data),
                          'lower': 0.016,
                          'upper': 0.0226},
                '-4.5': {'fraction': sum(1 for d in new_data if 4.5 <= d < 5) / len(new_data),
                          'lower': 0.0226,
                          'upper': 0.032},
                '-5': {'fraction': sum(1 for d in new_data if 5 <= d < 5.5) / len(new_data),
                          'lower': 0.032,
                          'upper': 0.045},
                '-5.5': {'fraction': sum(1 for d in new_data if 5.5 <= d < 6) / len(new_data),
                          'lower': 0.045,
                          'upper': 0.064},
                '-6': {'fraction': sum(1 for d in new_data if 6 <= d < 6.5) / len(new_data),
                           'lower': 0.064,
                           'upper': 0.091},
                '-6.5': {'fraction': sum(1 for d in new_data if 6.5 <= d < 7) / len(new_data),
                            'lower': 0.091,
                            'upper': 0.128},
                '-7': {'fraction': sum(1 for d in new_data if 7 <= d < 7.5) / len(new_data),
                            'lower': 0.128,
                            'upper': 181},
                '-7.5': {'fraction': sum(1 for d in new_data if 7.5 <= d < 8) / len(new_data),
                         'lower': 0.181,
                         'upper': 256},
                '-8': {'fraction': sum(1 for d in new_data if 8 <= d < 8.5) / len(new_data),
                       'lower': 256,
                       'upper': 362},
                '-8.5': {'fraction': sum(1 for d in new_data if 8.5 <= d < 9) / len(new_data),
                       'lower': 362,
                       'upper': 512},
                '-9': {'fraction': sum(1 for d in new_data if d >= 9) / len(new_data),
                      'lower': 512,
                      'upper': self.dn.loc[i, 'Dmax']}
            })

            # enforce minimum fraction
            if self.minimum_frac:
                counter = 0
                tot_added = 0
                ex_vals = [v['fraction'] for p, v in self.gs[i]['fractions'].items()]
                ex_vals.sort()
                for phi, vals in self.gs[i]['fractions'].items():
                    if float(phi) > -4 and vals['fraction'] < self.minimum_frac:
                        counter += 1
                        add = self.minimum_frac - vals['fraction']
                        tot_added += add
                        self.gs[i]['fractions'][phi]['fraction'] = vals['fraction'] + add
                    if float(phi) < -6 and 0 < vals['fraction'] < self.minimum_frac:
                        counter += 1
                        add = self.minimum_frac - vals['fraction']
                        tot_added += add
                        self.gs[i]['fractions'][phi]['fraction'] = vals['fraction'] + add
                if counter > 0:
                    subtract = tot_added / counter
                    vals_to_subtr = ex_vals[-counter:]
                    for phi, vals in self.gs[i]['fractions'].items():
                        if vals['fraction'] in vals_to_subtr:
                            self.gs[i]['fractions'][phi]['fraction'] = vals['fraction'] - subtract


            print(f'segment {i} D50: {2**np.percentile(new_data, 50)}')
            print(f'segment {i} D84: {2**np.percentile(new_data, 84)}')


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
    parser.add_argument('--da_field', help='The name of the field in the network attribute table containing values for'
                                         'upstream contributing drainage area', type=str, required=True)
    parser.add_argument('--max_size', help='A maximum grain size for all reaches in meters (default is 3 m)',
                        type=float, required=True, default=3)
    parser.add_argument('--min_sand', help='A minimum proportion of the bed distribution to assign to sand '
                                             'size classes (default is 0.01', type=float, required=True, default=0.01)

    args = parser.parse_args()

    GrainSize(args.stream_network, args.measurements, args.reach_ids, args.da_field, args.max_size, args.min_sand)


# if __name__ == '__main__':
#     main()

GrainSize(network='../Input_data/Woods_network_100m.shp', measurements=['../Input_data/woods_generated2.csv'], da_field='Drain_Area', reach_ids=[51], minimum_fraction=0.005)
