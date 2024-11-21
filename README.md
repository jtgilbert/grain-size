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
usage: predict_gsd.py [-h] [--measurements MEASUREMENTS [MEASUREMENTS ...]] [--slope_list SLOPE_LIST [SLOPE_LIST ...]] [--flow_list FLOW_LIST [FLOW_LIST ...]]
                      [--network NETWORK] [--slope_field SLOPE_FIELD] [--flow_field FLOW_FIELD] [--minimum_fraction MINIMUM_FRACTION] [--ID Field ID FIELD]

options:
  -h, --help            show this help message and exit
  --measurements MEASUREMENTS [MEASUREMENTS ...]
                        A list of paths to .csv files containing grain size measurements in mm with a header "D" for the column containing the measurements
  --slope_list SLOPE_LIST [SLOPE_LIST ...]
                        a list of slope values that corresponds with the measurement .csv files (in the same order)
  --flow_list FLOW_LIST [FLOW_LIST ...]
                        a list of unit flow or flow proxy values for the contributing basin of the measurement reach
  --network NETWORK     Path to a drainage network feature class/shapefile
  --slope_field SLOPE_FIELD
                        The title of the "slope" field in the drainage newtork feature class
  --flow_field FLOW_FIELD
                        The title of the "flow" or "flow proxy" field in the drainage network feature class
  --minimum_fraction MINIMUM_FRACTION
                        A minimum fraction to assign to half phi interval size classes
  --ID Field ID FIELD   A field from the drainage network to use as the label in the grain size distribution json file
```

for example, if my stream network was 'NHDPlus_Woods_Creek.shp', and I had two measurements, where the slope (field 'Slope') 
and flow proxy (field 'precip_da') values associated with measurement segments of the drainage network were 0.03 and 0.05 
for slope and 16 and 5 for the flow proxy, I would enter:

```commandline
python predict_gsd.py --measurements path_to_meas_1.csv path_to_meas_2.csv --slope_list 0.03 0.05 --flow_list 16 5 
--network /path/to/NHDPlus_Woods_Creek.shp --slope_field 'Slope' --flow_field 'precip_da' --minimum_fraction 0.005
```