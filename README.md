# grain-size
Estimate grain size distributions across a drainage network using field measurements.

## Setup
Download the .zip of the repository from the latest release. Navigate to the `scripts` folder, open a shell 
terminal, and run the bootstrap.sh script. This will set up a virtual environment with all of the necessary 
Python packages.

## Running the model
The simplest way to run the model is via command line. Open a command line terminal in the model folder and activate the 
python environment. (These instructions apply to UNIX systems, Windows still to come).

```commandline
source .venv/bin/activate
```
Then change directories into the `grain_size_distribution` folder, where the python scripts are located.

```commandline
cd grain_size_distribution
```

and run the model using the usage below.
```commandline
usage: grain_size.py [-h] --measurements MEASUREMENTS [MEASUREMENTS ...] --reach_ids REACH_IDS [REACH_IDS ...] --da_field DA_FIELD --max_size MAX_SIZE --min_fraction MIN_FRACTION stream_network

positional arguments:
  stream_network        Path to stream network feature class

optional arguments:
  -h, --help            show this help message and exit
  --measurements MEASUREMENTS [MEASUREMENTS ...]
                        A list of paths to csv files containing grain size measurements; should have header "D" at top of column followed by individual grain size measurements
  --reach_ids REACH_IDS [REACH_IDS ...]
                        A list containing the reach IDs from the stream network feature classassociated with the grain size measurements, in the same order as the measurements
  --da_field DA_FIELD   The name of the field in the network attribute table containing values forupstream contributing drainage area
  --max_size MAX_SIZE   A maximum grain size for all reaches in meters (default is 3 m)
  --min_fraction MIN_FRACTION
                        A minimum proportion of the bed distribution to assign to each grain size class (default is 0.005
```

for example, if my stream network was 'NHDPlus_Woods_Creek.shp', and I had two measurements,
associated with segments 46 and 68 of the drainage network, and the drainage area field was 
'Drain_Area' I would enter:

```commandline
python grain_size.py path_to_NHDPlus_Woods_Creek.shp --measurements path_to_meas_1.csv path_to_meas_2.csv --reach_ids 46 68 --da_field Drain_Area --max_size 2 --min_fraction 0.005
```