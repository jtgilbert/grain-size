import pandas as pd
import geopandas as gpd
import numpy as np
import argparse
import json
import os
import logging

from scipy import stats
from scipy.optimize import fmin
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression


class GrainSize:
    def __init__(self, measurements, slopes, precip_da, network: str, slope_field: str, precip_da_field: str,
                minimum_frac: float = None):

        self.measurements = measurements
        self.slopes = slopes
        self.precip_da = precip_da
        self.streams = network
        self.network = gpd.read_file(network)
        self.slope_field = slope_field
        self.precip_da_field = precip_da_field
        self.minimum_frac = minimum_frac

        logfile = os.path.join(os.path.dirname(self.measurements[0]), 'predict_gsd.log')
        logging.basicConfig(filename=logfile, format='%(levelname)s:%(message)s', level=logging.DEBUG)

        if self.slope_field not in self.network.columns:
            logging.info(f'No Slope field titled {slope_field} in drainage network attributes')
            raise Exception(f'No Slope field titled {slope_field} in drainage network attributes')
        if self.precip_da_field not in self.network.columns:
            logging.info(f'No Precipitation per area field titled {precip_da_field} in drainage network attributes')
            raise Exception(f'No Precipitation per area field titled {precip_da_field} in drainage network attributes')

        if len(self.measurements) > 1:
            params = self.calibration()
            gs_dict = self.attribute_network(params)
            gsd_out = self.generate_gsd(gs_dict=gs_dict)
            self.save_network()
            self.save_grain_size_dict(gsd_out)
        else:
            self.one_meas_stats()
            self.find_crit_depth()
            self.get_hydraulic_geom_coef()
            gs_dict = self.attribute_network_one_meas()
            gsd_out = self.generate_gsd(gs_dict=gs_dict)
            self.save_network()
            self.save_grain_size_dict(gsd_out)

    def calibration(self):
        calibration_stats = {'d50': {'low': [], 'mid': [], 'high':[]},
                             'd16': {'low': [], 'mid': [], 'high': []},
                             'd84': {'low': [], 'mid': [], 'high': []}}

        parameters_out = {}

        for i, meas in enumerate(self.measurements):
            df = pd.read_csv(meas)
            data = [np.log2(i) for i in list(df['D'])]
            a, loc, scale = stats.skewnorm.fit(data)

            # generate gsd to fill out distribution
            gsd = stats.skewnorm(a, loc, scale).rvs(1000)
            d50, d16, d84 = np.median(gsd), np.percentile(gsd, 16), np.percentile(gsd, 84)
            logging.info(f'{meas} stats -- D50: {int(2 ** d50)}, D16: {int(2 ** d16)}, D84: {int(2 ** d84)}')
            calibration_stats['d50']['mid'].append(d50)
            calibration_stats['d16']['mid'].append(d16)
            calibration_stats['d84']['mid'].append(d84)

            # bootstrap the data to get range
            num_boots = 100
            boot_medians = []
            boot_d16 = []
            boot_d84 = []
            for j in range(num_boots):
                boot = resample(gsd, replace=True, n_samples=len(0.8 * gsd))
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
            calibration_stats['d50']['low'].append(low_d50)
            calibration_stats['d50']['high'].append(high_d50)
            calibration_stats['d16']['low'].append(low_d16)
            calibration_stats['d16']['high'].append(high_d16)
            calibration_stats['d84']['low'].append(low_d84)
            calibration_stats['d84']['high'].append(high_d84)

        # check fit of linear vs log model
        lin_inputs = np.array([self.slopes[i] * self.precip_da[i] for i, s_val in enumerate(self.slopes)]).reshape(-1,1)
        log_inputs = np.array([np.log(self.slopes[i] * self.precip_da[i]) for i, s_val in enumerate(self.slopes)]).reshape(-1,1)
        lin_mod = LinearRegression()
        lin_mod.fit(lin_inputs, calibration_stats['d50']['mid'])
        lin_r2 = lin_mod.score(lin_inputs, calibration_stats['d50']['mid'])
        logging.info(f'Linear model r2: {lin_r2}')
        log_mod = LinearRegression()
        log_mod.fit(log_inputs, calibration_stats['d50']['mid'])
        log_r2 = log_mod.score(log_inputs, calibration_stats['d50']['mid'])
        logging.info(f'Log model r2: {log_r2}')
        if lin_r2 > log_r2:
            inputs = lin_inputs
            parameters_out['model_type'] = 'linear'
        else:
            inputs = log_inputs
            parameters_out['model_type'] = 'log'

        # get model parameters
        d50_mid_mod = LinearRegression()
        d50_mid_mod.fit(inputs, calibration_stats['d50']['mid'])
        parameters_out['d50_mid'] = {'coefs': d50_mid_mod.coef_, 'intercept': d50_mid_mod.intercept_}
        logging.info(f'D50 mid regression fit: {d50_mid_mod.score(inputs, calibration_stats["d50"]["mid"])}')
        print(f'D50 mid regression fit: {d50_mid_mod.score(inputs, calibration_stats["d50"]["mid"])}')
        d50_low_mod = LinearRegression()
        d50_low_mod.fit(inputs, calibration_stats['d50']['low'])
        parameters_out['d50_low'] = {'coefs': d50_low_mod.coef_, 'intercept': d50_low_mod.intercept_}
        logging.info(f'D50 low regression fit: {d50_low_mod.score(inputs, calibration_stats["d50"]["low"])}')
        d50_high_mod = LinearRegression()
        d50_high_mod.fit(inputs, calibration_stats['d50']['high'])
        parameters_out['d50_high'] = {'coefs': d50_high_mod.coef_, 'intercept': d50_high_mod.intercept_}
        logging.info(f'D50 high regression fit: {d50_high_mod.score(inputs, calibration_stats["d50"]["high"])}')

        d16_mid_mod = LinearRegression()
        d16_mid_mod.fit(inputs, calibration_stats['d16']['mid'])
        parameters_out['d16_mid'] = {'coefs': d16_mid_mod.coef_, 'intercept': d16_mid_mod.intercept_}
        logging.info(
            f'D16 mid regression fit: {d16_mid_mod.score(inputs, calibration_stats["d16"]["mid"])}')
        d16_low_mod = LinearRegression()
        d16_low_mod.fit(inputs, calibration_stats['d16']['low'])
        parameters_out['d16_low'] = {'coefs': d16_low_mod.coef_, 'intercept': d16_low_mod.intercept_}
        logging.info(
            f'D16 low regression fit: {d16_low_mod.score(inputs, calibration_stats["d16"]["low"])}')
        d16_high_mod = LinearRegression()
        d16_high_mod.fit(inputs, calibration_stats['d16']['high'])
        parameters_out['d16_high'] = {'coefs': d16_high_mod.coef_, 'intercept': d16_high_mod.intercept_}
        logging.info(
            f'D16 high regression fit: {d16_high_mod.score(inputs, calibration_stats["d16"]["high"])}')

        d84_mid_mod = LinearRegression()
        d84_mid_mod.fit(inputs, calibration_stats['d84']['mid'])
        parameters_out['d84_mid'] = {'coefs': d84_mid_mod.coef_, 'intercept': d84_mid_mod.intercept_}
        logging.info(
            f'D84 mid regression fit: {d84_mid_mod.score(inputs, calibration_stats["d84"]["mid"])}')
        d84_low_mod = LinearRegression()
        d84_low_mod.fit(inputs, calibration_stats['d84']['low'])
        parameters_out['d84_low'] = {'coefs': d84_low_mod.coef_, 'intercept': d84_low_mod.intercept_}
        logging.info(
            f'D84 low regression fit: {d84_low_mod.score(inputs, calibration_stats["d84"]["low"])}')
        d84_high_mod = LinearRegression()
        d84_high_mod.fit(inputs, calibration_stats['d84']['high'])
        parameters_out['d84_high'] = {'coefs': d84_high_mod.coef_, 'intercept': d84_high_mod.intercept_}
        logging.info(
            f'D84 high regression fit: {d84_high_mod.score(inputs, calibration_stats["d84"]["high"])}')

        return parameters_out

    def attribute_network(self, regression_params):

        stats_dict = {}

        for i in self.network.index:
            if regression_params['model_type'] == 'linear':
                d50 = 2 ** ((self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                            regression_params['d50_mid']['coefs'] + regression_params['d50_mid']['intercept'])
                self.network.loc[i, 'D50'] = d50
                d50_low = 2 ** ((self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                regression_params['d50_low']['coefs'] + regression_params['d50_low']['intercept'])
                self.network.loc[i, 'D50_low'] = d50_low
                d50_high = 2 ** ((self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                 regression_params['d50_high']['coefs'] + regression_params['d50_high']['intercept'])
                self.network.loc[i, 'D50_high'] = d50_high
                d16 = 2 ** ((self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                            regression_params['d16_mid']['coefs'] + regression_params['d16_mid']['intercept'])
                self.network.loc[i, 'D16'] = d16
                d16_low = 2 ** ((self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                regression_params['d16_low']['coefs'] + regression_params['d16_low']['intercept'])
                self.network.loc[i, 'D16_low'] = d16_low
                d16_high = 2 ** ((self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                 regression_params['d16_high']['coefs'] + regression_params['d16_high']['intercept'])
                self.network.loc[i, 'D16_high'] = d16_high
                d84 = 2 ** ((self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                            regression_params['d84_mid']['coefs'] + regression_params['d84_mid']['intercept'])
                self.network.loc[i, 'D84'] = d84
                d84_low = 2 ** ((self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                regression_params['d84_low']['coefs'] + regression_params['d84_low']['intercept'])
                self.network.loc[i, 'D84_low'] = d84_low
                d84_high = 2 ** ((self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                 regression_params['d84_high']['coefs'] + regression_params['d84_high']['intercept'])
                self.network.loc[i, 'D84_high'] = d84_high
            else:
                d50 = 2 ** (np.log(self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                            regression_params['d50_mid']['coefs'] + regression_params['d50_mid']['intercept'])
                self.network.loc[i, 'D50'] = d50
                d50_low = 2 ** (np.log(self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                regression_params['d50_low']['coefs'] + regression_params['d50_low']['intercept'])
                self.network.loc[i, 'D50_low'] = d50_low
                d50_high = 2 ** (np.log(self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                 regression_params['d50_high']['coefs'] + regression_params['d50_high']['intercept'])
                self.network.loc[i, 'D50_high'] = d50_high
                d16 = 2 ** (np.log(self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                            regression_params['d16_mid']['coefs'] + regression_params['d16_mid']['intercept'])
                self.network.loc[i, 'D16'] = d16
                d16_low = 2 ** (np.log(self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                regression_params['d16_low']['coefs'] + regression_params['d16_low']['intercept'])
                self.network.loc[i, 'D16_low'] = d16_low
                d16_high = 2 ** (np.log(self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                 regression_params['d16_high']['coefs'] + regression_params['d16_high']['intercept'])
                self.network.loc[i, 'D16_high'] = d16_high
                d84 = 2 ** (np.log(self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                            regression_params['d84_mid']['coefs'] + regression_params['d84_mid']['intercept'])
                self.network.loc[i, 'D84'] = d84
                d84_low = 2 ** (np.log(self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                regression_params['d84_low']['coefs'] + regression_params['d84_low']['intercept'])
                self.network.loc[i, 'D84_low'] = d84_low
                d84_high = 2 ** (np.log(self.network.loc[i, self.slope_field] * self.network.loc[i, self.precip_da_field]) *
                                 regression_params['d84_high']['coefs'] + regression_params['d84_high']['intercept'])
                self.network.loc[i, 'D84_high'] = d84_high

            stats_dict[i] = {
                'd50': d50[0],
                'd50_low': d50_low[0],
                'd50_high': d50_high[0],
                'd16': d16[0],
                'd16_low': d16_low[0],
                'd16_high': d16_high[0],
                'd84': d84[0],
                'd84_low': d84_low[0],
                'd84_high': d84_high[0],
                'fractions': {}
            }

        return stats_dict

    def one_meas_stats(self):

        df = pd.read_csv(self.measurements[0])
        data = [np.log2(i / 1000) for i in list(df['D'])]
        a, loc, scale = stats.skewnorm.fit(data)

        # generate gsd to fill out distribution
        gsd = stats.skewnorm(a, loc, scale).rvs(1000)
        d50, d16, d84 = np.median(gsd), np.percentile(gsd, 16), np.percentile(gsd, 84)

        # bootstrap the data to get range
        num_boots = 100
        boot_medians = []
        boot_d16 = []
        boot_d84 = []
        for j in range(num_boots):
            boot = resample(gsd, replace=True, n_samples=len(0.8 * gsd))
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

        self.stats = {
            'slope': self.slopes[0],
            'precip': self.precip_da[0],
            'd16': d16,
            'd16_low': low_d16,
            'd16_high': high_d16,
            'd50': d50,
            'd50_low': low_d50,
            'd50_high': high_d50,
            'd84': d84,
            'd84_low': low_d84,
            'd84_high': high_d84
        }

        return self.stats

    def find_crit_depth(self):

        rho_s = 2650  # density of sediment
        rho = 1000  # density of water

        roughness = self.stats['d84'] / self.stats['d50']
        if roughness <= 2:
            tau_coef = 0.029
        else:
            tau_coef = 0.043 * np.log(roughness) - 0.0005
        if tau_coef < 0.029:
            tau_coef = 0.029
        tau_c_star_16 = tau_coef * (self.stats['d16'] / self.stats['d50']) ** -0.68
        tau_c_16 = tau_c_star_16 * (rho_s - rho) * 9.81 * self.stats['d16']
        h_c_16 = tau_c_16 / (rho * 9.81 * self.stats['slope'])
        tau_c_16_low = tau_c_star_16 * (rho_s - rho) * 9.81 * self.stats['d16_low']
        h_c_16_low = tau_c_16_low / (rho * 9.81 * self.stats['slope'])
        tau_c_16_high = tau_c_star_16 * (rho_s - rho) * 9.81 * self.stats['d16_high']
        h_c_16_high = tau_c_16_high / (rho * 9.81 * self.stats['slope'])
        tau_c_star_50 = tau_coef
        tau_c_50 = tau_c_star_50 * (rho_s - rho) * 9.81 * self.stats['d50']
        h_c_50 = tau_c_50 / (rho * 9.81 * self.stats['slope'])
        tau_c_50_low = tau_c_star_50 * (rho_s - rho) * 9.81 * self.stats['d50_low']
        h_c_50_low = tau_c_50_low / (rho * 9.81 * self.stats['slope'])
        tau_c_50_high = tau_c_star_50 * (rho_s - rho) * 9.81 * self.stats['d50_high']
        h_c_50_high = tau_c_50_high / (rho * 9.81 * self.stats['slope'])
        tau_c_star_84 = tau_coef * (self.stats['d84'] / self.stats['d50']) ** -0.68
        tau_c_84 = tau_c_star_84 * (rho_s - rho) * 9.81 * self.stats['d84']
        h_c_84 = tau_c_84 / (rho * 9.81 * self.stats['slope'])
        tau_c_84_low = tau_c_star_84 * (rho_s - rho) * 9.81 * self.stats['d84_low']
        h_c_84_low = tau_c_84_low / (rho * 9.81 * self.stats['slope'])
        tau_c_84_high = tau_c_star_84 * (rho_s - rho) * 9.81 * self.stats['d84_high']
        h_c_84_high = tau_c_84_high / (rho * 9.81 * self.stats['slope'])
        self.stats.update({'depth_d16': h_c_16,
                           'depth_d16_low': h_c_16_low,
                           'depth_d16_high': h_c_16_high,
                           'depth_d50': h_c_50,
                           'depth_d50_low': h_c_50_low,
                           'depth_d50_high': h_c_50_high,
                           'depth_d84': h_c_84,
                           'depth_d84_low': h_c_84_low,
                           'depth_d84_high': h_c_84_high})

    def get_hydraulic_geom_coef(self):

        self.stats.update({'coefs':
                               {'h_coef_16': self.stats['depth_d16'] / self.stats['precip'] ** 0.4,
            'h_coef_16_low': self.stats['depth_d16_low'] / self.stats['precip'] ** 0.4,
            'h_coef_16_high': self.stats['depth_d16_high'] / self.stats['precip'] ** 0.4,
            'h_coef_50': self.stats['depth_d50'] / self.stats['precip'] ** 0.4,
            'h_coef_50_low': self.stats['depth_d50_low'] / self.stats['precip'] ** 0.4,
            'h_coef_50_high': self.stats['depth_d50_high'] / self.stats['precip'] ** 0.4,
            'h_coef_84': self.stats['depth_d84'] / self.stats['precip'] ** 0.4,
            'h_coef_84_low': self.stats['depth_d84_low'] / self.stats['precip'] ** 0.4,
            'h_coef_84_high': self.stats['depth_d84_high'] / self.stats['precip'] ** 0.4}
        })

    def attribute_network_one_meas(self):

        stats_dict = {}
        for i in self.network.index:
            print(f'Estimating D50, D16, and D84 for segment: {i}')
            if self.stats['d84'] / self.stats['d50'] < 2:
                tau_coef = 0.029
            else:
                tau_coef = 0.043 * np.log(self.stats['d84'] / self.stats['d50']) - 0.0005
            if tau_coef < 0.029:
                tau_coef = 0.029

            h_c_50 = self.stats['coefs']['h_coef_50'] * self.network.loc[i, self.precip_da_field] ** 0.4
            tau_c_50 = 1000 * 9.81 * h_c_50 * self.network.loc[i, self.slope_field]
            d50 = tau_c_50 / ((2650 - 1000) * 9.81 * tau_coef)
            self.network.loc[i, 'D50'] = d50

            h_c_50_low = self.stats['coefs']['h_coef_50_low'] * self.network.loc[i, self.precip_da_field] ** 0.4
            tau_c_50_low = 1000 * 9.81 * h_c_50_low * self.network.loc[i, self.slope_field]
            d50_low = tau_c_50_low / ((2650 - 1000) * 9.81 * tau_coef)
            self.network.loc[i, 'D50_low'] = d50_low

            h_c_50_high = self.stats['coefs']['h_coef_50_high'] * self.network.loc[i, self.precip_da_field] ** 0.4
            tau_c_50_high = 1000 * 9.81 * h_c_50_high * self.network.loc[i, self.slope_field]
            d50_high = tau_c_50_high / ((2650 - 1000) * 9.81 * tau_coef)
            self.network.loc[i, 'D50_high'] = d50_high

            h_c_16 = self.stats['coefs']['h_coef_16'] * self.network.loc[i, self.precip_da_field] ** 0.4
            tau_c_16 = 1000 * 9.81 * h_c_16 * self.network.loc[i, self.slope_field]
            d16 = tau_c_16 / ((2650 - 1000) * 9.81 * tau_coef)
            self.network.loc[i, 'D16'] = d16

            h_c_16_low = self.stats['coefs']['h_coef_16_low'] * self.network.loc[i, self.precip_da_field] ** 0.4
            tau_c_16_low = 1000 * 9.81 * h_c_16_low * self.network.loc[i, self.slope_field]
            d16_low = tau_c_16_low / ((2650 - 1000) * 9.81 * tau_coef)
            self.network.loc[i, 'D16_low'] = d16_low

            h_c_16_high = self.stats['coefs']['h_coef_16_high'] * self.network.loc[i, self.precip_da_field] ** 0.4
            tau_c_16_high = 1000 * 9.81 * h_c_16_high * self.network.loc[i, self.slope_field]
            d16_high = tau_c_16_high / ((2650 - 1000) * 9.81 * tau_coef)
            self.network.loc[i, 'D16_high'] = d16_high

            h_c_84 = self.stats['coefs']['h_coef_84'] * self.network.loc[i, self.precip_da_field] ** 0.4
            tau_c_84 = 1000 * 9.81 * h_c_84 * self.network.loc[i, self.slope_field]
            d84 = tau_c_84 / ((2650 - 1000) * 9.81 * tau_coef)
            self.network.loc[i, 'D84'] = d84

            h_c_84_low = self.stats['coefs']['h_coef_84_low'] * self.network.loc[i, self.precip_da_field] ** 0.4
            tau_c_84_low = 1000 * 9.81 * h_c_84_low * self.network.loc[i, self.slope_field]
            d84_low = tau_c_84_low / ((2650 - 1000) * 9.81 * tau_coef)
            self.network.loc[i, 'D84_low'] = d84_low

            h_c_84_high = self.stats['coefs']['h_coef_84_high'] * self.network.loc[i, self.precip_da_field] ** 0.4
            tau_c_84_high = 1000 * 9.81 * h_c_84_high * self.network.loc[i, self.slope_field]
            d84_high = tau_c_84_high / ((2650 - 1000) * 9.81 * tau_coef)
            self.network.loc[i, 'D84_high'] = d84_high

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

    def generate_gsd(self, gs_dict):

        def loc_err(loc, a, scale, d50):
            dist = stats.skewnorm(a, loc, scale).rvs(10000)
            d50_est = np.percentile(dist, 50)
            err = (d50 - d50_est) ** 2

            return err

        def scale_err(scale, a, loc, d16, d84):
            if scale <= 0:
                scale = abs(scale)
            dist = stats.skewnorm(a, loc, scale).rvs(10000)
            err = (np.percentile(dist, 16) - d16) ** 2 + (np.percentile(dist, 84) - d84) ** 2

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

        for i in self.network.index:
            print(f'finding distribution for segment {i}')
            d_50, d_16, d_84 = np.log2(self.network.loc[i, 'D50']), np.log2(self.network.loc[i, 'D16']), np.log2(self.network.loc[i, 'D84'])

            res = fmin(loc_err, loc, args=(a, scale, d_50))
            loc_opt = res[0]

            res2 = fmin(scale_err, scale, args=(a, loc_opt, d_16, d_84))
            scale_opt = res2[0]
            # while scale_opt <= 0:
            #    res2 = fmin(scale_err, abs(scale_opt), args=(a, loc_opt, d_50))
            #    scale_opt = res2[0]

            new_data = stats.skewnorm(a, loc_opt, scale_opt).rvs(10000)
            new_data = new_data[new_data >= -1]

            gs_dict[i]['fractions'].update({
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
                       'upper': None}
            })

            # enforce minimum fraction
            if self.minimum_frac:
                counter = 0
                tot_added = 0
                ex_vals = [v['fraction'] for p, v in gs_dict[i]['fractions'].items()]
                ex_vals.sort()
                for phi, vals in gs_dict[i]['fractions'].items():
                    if float(phi) > -4 and vals['fraction'] < self.minimum_frac:
                        counter += 1
                        add = self.minimum_frac - vals['fraction']
                        tot_added += add
                        gs_dict[i]['fractions'][phi]['fraction'] = vals['fraction'] + add
                    if float(phi) < -6 and 0 < vals['fraction'] < self.minimum_frac:
                        counter += 1
                        add = self.minimum_frac - vals['fraction']
                        tot_added += add
                        gs_dict[i]['fractions'][phi]['fraction'] = vals['fraction'] + add
                if counter > 0:
                    subtract = tot_added / counter
                    vals_to_subtr = ex_vals[-counter:]
                    for phi, vals in gs_dict[i]['fractions'].items():
                        if vals['fraction'] in vals_to_subtr:
                            gs_dict[i]['fractions'][phi]['fraction'] = vals['fraction'] - subtract

            print(f'segment {i} D50: {2 ** np.percentile(new_data, 50)}')
            print(f'segment {i} D84: {2 ** np.percentile(new_data, 84)}')

        return gs_dict

    def save_network(self):
        self.network.to_file(self.streams)

    def save_grain_size_dict(self, gsd_dict):
        with open(os.path.join(os.path.dirname(self.streams), 'gsd.json'), 'w') as f:
            json.dump(gsd_dict, f, indent=4)


# def main():
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('measurements', help='A list of paths to .csv files containing grain size measurements in mm '
#                                             'with a header "D" for the column containing the measurements', nargs='+',
#                         type=str, required=True)
#     parser.add_argument('slope_list', help='a list of slope values that corresponds with the measurement .csv files ('
#                                            'in the same order)', )
#     parser.add_argument('unit_precip_list', help='')
#     parser.add_argument('network', help='')
#     parser.add_argument('slope_field', help='')
#     parser.add_argument('unit_precip_field', help='')
#     parser.add_argument('minimum_fraction', help='')
#
#     args = parser.parse_args()
#
#     GrainSize(args.measurements, args.slope_list, args.unit_precip_list, args.network, args.slope_field,
#               args.unit_precip_field, args.minimum_fraction)


GrainSize(measurements=['../Input_data/Deer_Cr/D_Deer_Creek_lower.csv',
                        '../Input_data/Deer_Cr/D_Deer_Creek_trib.csv',
                        '../Input_data/Deer_Cr/D_Deer_Creek_upper.csv',
                        '../Input_data/Deer_Cr/D_Woods_headwater.csv'],
          slopes=[0.01, 0.08, 0.021, 0.037], precip_da=[9.07, 0.058, 8.56, 0.214],
          network='../Input_data/Woods_network_100m.shp', slope_field='Slope', precip_da_field='unit_preci')


