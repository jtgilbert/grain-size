import geopandas as gpd
import pandas as pd
import numpy as np


class GrainSizeFraction:
    """
    Estimates fraction of the bed material in each grain size class for every segment of a drainage network
    using one or more field measurements
    """

    def __init__(self, network, measurements, reach_ids):

        if len(measurements) != len(reach_ids):
            raise Exception('Different number of measurements and associated reach IDs entered')

        self.streams = network
        self.reach_ids = reach_ids
        self.dn = gpd.read_file(network)

        self.reach_stats = {}
        for i, meas in enumerate(measurements):
            df = pd.read_csv(meas)
            df['D'] = df['D']/1000  # converting from mm to m

            self.reach_stats[reach_ids[i]] = {
                'slope': self.dn.loc[reach_ids[i], 'Slope'],
                'drain_area': self.dn.loc[reach_ids[i], 'TotDASqKm'],
                'd50': np.median(df['D']),
                'sizeFractions': {
                    '0-1': {'fraction': sum(1 for d in df['D'] if 0 < d <= 0.001) / len(df['D']),
                            'upper': 0.001},
                    '1-2': {'fraction': sum(1 for d in df['D'] if 0.001 < d <= 0.002) / len(df['D']),
                            'upper': 0.002},
                    '2-4': {'fraction': sum(1 for d in df['D'] if 0.002 < d <= 0.004) / len(df['D']),
                            'upper': 0.004},
                    '4-6': {'fraction': sum(1 for d in df['D'] if 0.004 < d <= 0.006) / len(df['D']),
                            'upper': 0.006},
                    '6-8': {'fraction': sum(1 for d in df['D'] if 0.006 < d <= 0.008) / len(df['D']),
                            'upper': 0.008},
                    '8-12': {'fraction': sum(1 for d in df['D'] if 0.008 < d <= 0.012) / len(df['D']),
                             'upper': 0.012},
                    '12-16': {'fraction': sum(1 for d in df['D'] if 0.012 < d <= 0.016) / len(df['D']),
                              'upper': 0.016},
                    '16-24': {'fraction': sum(1 for d in df['D'] if 0.016 < d <= 0.024) / len(df['D']),
                              'upper': 0.024},
                    '24-32': {'fraction': sum(1 for d in df['D'] if 0.024 < d <= 0.032) / len(df['D']),
                              'upper': 0.032},
                    '32-48': {'fraction': sum(1 for d in df['D'] if 0.032 < d <= 0.048) / len(df['D']),
                              'upper': 0.048},
                    '48-64': {'fraction': sum(1 for d in df['D'] if 0.048 < d <= 0.064) / len(df['D']),
                              'upper': 0.064},
                    '64-96': {'fraction': sum(1 for d in df['D'] if 0.064 < d <= 0.096) / len(df['D']),
                              'upper': 0.096},
                    '96-128': {'fraction': sum(1 for d in df['D'] if 0.096 < d <= 0.128) / len(df['D']),
                               'upper': 0.128},
                    '128-192': {'fraction': sum(1 for d in df['D'] if 0.128 < d <= 0.192) / len(df['D']),
                                'upper': 0.192},
                    '192-256': {'fraction': sum(1 for d in df['D'] if 0.192 < d <= 0.256) / len(df['D']),
                                'upper': 256},
                    '>256': {'fraction': sum(1 for d in df['D'] if 0.256 < d) / len(df['D']),
                             'upper': 512}
                }
            }

        self.find_crit_depth()

    def find_crit_depth(self):

        for reachid, vals in self.reach_stats.items():

            tau_c_star_50 = 0.038
            tau_c_50 = tau_c_star_50 * 1650 * 9.81 * vals['d50']
            h_c_50 = tau_c_50 / (1000 * 9.81 * vals['slope'])
            self.reach_stats[reachid].update({'critDepth50': h_c_50})

            for key in vals['sizeFractions'].keys():
                tau_c_star = 0.038*(vals['sizeFractions'][key]['upper']/vals['d50'])**-0.65
                tau_c = tau_c_star * 1650 * 9.81 * vals['sizeFractions'][key]['upper']
                h_c = tau_c / (1000 * 9.81 * vals['slope'])
                self.reach_stats[reachid]['sizeFractions'][key].update({'critDepth': h_c})

        print('stop')

if __name__ == '__main__':

    GrainSizeFraction(network='./Input_data/NHDPlus_Woods.shp', measurements=['./Input_data/Woods_D.csv'], reach_ids=[41])