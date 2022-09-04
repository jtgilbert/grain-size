import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.utils import resample


class GrainSize:

    def __init__(self, network, measurements, reach_ids):

        if len(measurements) != len(reach_ids):
            raise Exception('Different number of measurements and associated reach IDs entered')

        self.streams = network
        self.reach_ids = reach_ids
        self.dn = gpd.read_file(network)

        self.reach_stats = {}  # {reach: {slope: val, drain_area: val, d16: val, d50: val, d84: val}, reach: {}}
        for i, meas in enumerate(measurements):
            df = pd.read_csv(meas)
            df['D'] = df['D']/1000  # converting from mm to m

            # bootstrap the data to get range
            num_boots = 100
            boot_medians = []
            for j in range(num_boots):
                boot = resample(df['D'], replace=True, n_samples=len(0.8*df['D']))
                median = np.median(boot)
                boot_medians.append(median)
            medgr = np.asarray(boot_medians)
            mean_d = np.mean(medgr)
            low_d50 = mean_d - (1.64 * np.std(medgr))  # does there need to be a min
            high_d50 = mean_d + (1.64 * np.std(medgr))

            self.reach_stats[reach_ids[i]] = {
                'slope': self.dn.loc[reach_ids[i], 'Slope'],
                'drain_area': self.dn.loc[reach_ids[i], 'TotDASqKm'],  # might need to change to DivDA
                'd16': np.percentile(df['D'], 16),
                'd50': np.median(df['D']),
                'd50_low': low_d50,
                'd50_high': high_d50,
                'd84': np.percentile(df['D'], 84)
            }

        self.find_crit_depth()
        self.get_hydraulic_geom_coef()
        self.estimate_grain_size()

        print('stop')

    def find_crit_depth(self):  # for now only option is using my critical shields method; could addd fixed eg 0.045

        rho_s = 2650  # density of sediment
        rho = 1000  # density of water

        for reachid, vals in self.reach_stats.items():

            tau_c_star_16 = 0.038*(vals['d16']/vals['d50'])**-0.65
            tau_c_16 = tau_c_star_16 * (rho_s-rho) * 9.81 * vals['d16']
            h_c_16 = tau_c_16 / (rho * 9.81 * vals['slope'])
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
            self.reach_stats[reachid].update({'depth_d16': h_c_16,
                                              'depth_d50': h_c_50,
                                              'depth_d50_low': h_c_50_low,
                                              'depth_d50_high': h_c_50_high,
                                              'depth_d84': h_c_84})

    # if there's only one measurement, use default hydraulic geometry
    def get_hydraulic_geom_coef(self):

        for reachid, vals in self.reach_stats.items():

            self.reach_stats[reachid].update({
                'h_coef_16': self.reach_stats[reachid]['depth_d16']/self.reach_stats[reachid]['drain_area']**0.4,
                'h_coef_50': self.reach_stats[reachid]['depth_d50']/self.reach_stats[reachid]['drain_area']**0.4,
                'h_coef_50_low': self.reach_stats[reachid]['depth_d50_low'] / self.reach_stats[reachid]['drain_area'] ** 0.4,
                'h_coef_50_high': self.reach_stats[reachid]['depth_d50_high'] / self.reach_stats[reachid]['drain_area'] ** 0.4,
                'h_coef_84': self.reach_stats[reachid]['depth_d84']/self.reach_stats[reachid]['drain_area']**0.4
            })

    # if there's more than one measurement refine hydraulic geometry relation for basin

    def estimate_grain_size(self):

        if len(self.reach_stats) == 1:
            for i in self.dn.index:
                print(f'Estimating D50 for segment : {i}')
                h_c_50 = self.reach_stats[self.reach_ids[0]]['h_coef_50']*self.dn.loc[i, 'TotDASqKm']**0.4
                tau_c_50 = 1000 * 9.81 * h_c_50 * self.dn.loc[i, 'Slope']
                d50 = tau_c_50 / ((2650 - 1000) * 9.81 * 0.038)
                self.dn.loc[i, 'D50_est'] = d50 * 1000

                h_c_50_low = self.reach_stats[self.reach_ids[0]]['h_coef_50_low'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
                tau_c_50_low = 1000 * 9.81 * h_c_50_low * self.dn.loc[i, 'Slope']
                d50_low = tau_c_50_low / ((2650 - 1000) * 9.81 * 0.038)
                self.dn.loc[i, 'D50_low_est'] = d50_low * 1000

                h_c_50_high = self.reach_stats[self.reach_ids[0]]['h_coef_50_high'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
                tau_c_50_high = 1000 * 9.81 * h_c_50_high * self.dn.loc[i, 'Slope']
                d50_high = tau_c_50_high / ((2650 - 1000) * 9.81 * 0.038)
                self.dn.loc[i, 'D50_high_est'] = d50_high * 1000

                h_c_16 = self.reach_stats[self.reach_ids[0]]['h_coef_16']*self.dn.loc[i, 'TotDASqKm']**0.4
                tau_c_16 = 1000 * 9.81 * h_c_16 * self.dn.loc[i, 'Slope']
                tau_c_star_16 = 0.038 * (self.reach_stats[self.reach_ids[0]]['d16']/self.reach_stats[self.reach_ids[0]]['d50']) ** -0.65
                d16 = tau_c_16 / ((2650 - 1000) * 9.81 * tau_c_star_16)
                self.dn.loc[i, 'D16_est'] = d16 * 1000

                h_c_84 = self.reach_stats[self.reach_ids[0]]['h_coef_84'] * self.dn.loc[i, 'TotDASqKm'] ** 0.4
                tau_c_84 = 1000 * 9.81 * h_c_84 * self.dn.loc[i, 'Slope']
                tau_c_star_84 = 0.038 * (self.reach_stats[self.reach_ids[0]]['d84'] / self.reach_stats[self.reach_ids[0]]['d50']) ** -0.65
                d84 = tau_c_84 / ((2650 - 1000) * 9.81 * tau_c_star_84)
                self.dn.loc[i, 'D84_est'] = d84 * 1000

        self.dn.to_file(self.streams)


if __name__ == '__main__':

    GrainSize(network='./Input_data/NHDPlus_Woods.shp', measurements=['./Input_data/Woods_D.csv'], reach_ids=[41])
