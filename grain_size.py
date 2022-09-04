import geopandas as gpd
import numpy as np
import pandas as pd


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
            self.reach_stats[reach_ids[i]] = {
                'slope': self.dn.loc[reach_ids[i], 'Slope'],
                'drain_area': self.dn.loc[reach_ids[i], 'TotDASqKm'],  # might need to change to DivDA
                'd16': np.percentile(df['D'], 16),
                'd50': np.median(df['D']),
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
            tau_c_star_84 = 0.038 * (vals['d84'] / vals['d50']) ** -0.65
            tau_c_84 = tau_c_star_84 * (rho_s - rho) * 9.81 * vals['d84']
            h_c_84 = tau_c_84 / (rho * 9.81 * vals['slope'])
            self.reach_stats[reachid].update({'depth_d16': h_c_16, 'depth_d50': h_c_50, 'depth_d84': h_c_84})

    # if there's only one measurement, use default hydraulic geometry
    def get_hydraulic_geom_coef(self):

        for reachid, vals in self.reach_stats.items():

            self.reach_stats[reachid].update({
                'h_coef_16': self.reach_stats[reachid]['depth_d16']/self.reach_stats[reachid]['drain_area']**0.4,
                'h_coef_50': self.reach_stats[reachid]['depth_d50']/self.reach_stats[reachid]['drain_area']**0.4,
                'h_coef_84': self.reach_stats[reachid]['depth_d84']/self.reach_stats[reachid]['drain_area']**0.4
            })

    # if there's more than one measurement refine hydraulic geometry relation for basin

    def estimate_grain_size(self):

        if len(self.reach_stats) == 1:
            for i in self.dn.index:
                print(f'Estimating D50 for segment : {i}')
                h_c_50 = self.reach_stats[self.reach_ids[0]]['h_coef_50']*self.dn.loc[i, 'TotDASqKm']**0.4
                print(f'critical depth: {h_c_50}')
                tau_c_50 = 1000 * 9.81 * h_c_50 * self.dn.loc[i, 'Slope']
                d50 = tau_c_50 / ((2650 - 1000) * 9.81 * 0.038)
                self.dn.loc[i, 'D50_est'] = d50 * 1000

        self.dn.to_file(self.streams)


if __name__ == '__main__':

    GrainSize(network='./Input_data/NHDPlus_Woods.shp', measurements=['./Input_data/Woods_D.csv'], reach_ids=[41])
