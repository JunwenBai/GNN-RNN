import geopandas as gpd
import os
import pandas as pd
import matplotlib.pyplot as plt

MAP_FILE = "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/gz_2010_us_050_00_20m/"

def plot_county_data(counties, values, column_name, year,
                     output_dir='/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/figures/python'):
    
    # Create data frame mapping county FIPS to attribute value
    data_table = pd.DataFrame({'fips': counties, column_name: values})
    print("Plotting county data")
    print(data_table.head())

    # Read county map from file, and compute county FIPS
    county_map = gpd.read_file(MAP_FILE)
    county_map['fips'] = county_map['STATE'].astype(int) * 1000 + county_map['COUNTY'].astype(int)
    print(county_map.head())
    assert(len(set(counties)) == len(counties))  # Make sure no duplicate county entries

    # Add the variable values to the county map
    county_map = county_map.merge(data_table, on='fips')
    fig, ax = plt.subplots(1, 1, figsize=(45, 30))
    county_map.plot(column=column_name, ax=ax, legend=True,
                    missing_kwds={  # Red hatch for missing values. See https://geopandas.org/docs/user_guide/mapping.html#choropleth-maps
                        "color": "lightgrey",
                        "edgecolor": "red",
                        "hatch": "///",
                        "label": "Missing values",
                    })
    ax.set_title(column_name + ": " + str(year), fontdict={'fontsize': 30})
    fig.axes[0].tick_params(labelsize=24)
    fig.axes[1].tick_params(labelsize=24)
    plt.savefig(os.path.join(output_dir, column_name + "_" + str(year) + ".png"))
    plt.close()
