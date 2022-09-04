import geopandas as gpd
import pandas as pd


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
                '0-1': sum(1 for d in df['D'] if 0 < d <= 0.001) / len(df['D']),
                '1-2': sum(1 for d in df['D'] if 0.001 < d <= 0.002) / len(df['D']),
                '2-4': sum(1 for d in df['D'] if 0.002 < d <= 0.004) / len(df['D']),
                '4-6': sum(1 for d in df['D'] if 0.004 < d <= 0.006) / len(df['D']),
                '6-8': sum(1 for d in df['D'] if 0.006 < d <= 0.008) / len(df['D']),
                '8-12': sum(1 for d in df['D'] if 0.008 < d <= 0.012) / len(df['D']),
                '12-16': sum(1 for d in df['D'] if 0.012 < d <= 0.016) / len(df['D']),
                '16-24': sum(1 for d in df['D'] if 0.016 < d <= 0.024) / len(df['D']),
                '24-32': sum(1 for d in df['D'] if 0.024 < d <= 0.032) / len(df['D']),
                '32-48': sum(1 for d in df['D'] if 0.032 < d <= 0.048) / len(df['D']),
                '48-64': sum(1 for d in df['D'] if 0.048 < d <= 0.064) / len(df['D']),
                '64-96': sum(1 for d in df['D'] if 0.064 < d <= 0.096) / len(df['D']),
                '96-128': sum(1 for d in df['D'] if 0.096 < d <= 0.128) / len(df['D']),
                '128-192': sum(1 for d in df['D'] if 0.128 < d <= 0.192) / len(df['D']),\
                '192-256': sum(1 for d in df['D'] if 0.192 < d <= 0.256) / len(df['D']),
                '>256': sum(1 for d in df['D'] if 0.256 < d) / len(df['D']),
            }

        print('stop')

if __name__ == '__main__':

    GrainSizeFraction(network='./Input_data/NHDPlus_Woods.shp', measurements=['./Input_data/Woods_D.csv'], reach_ids=[41])