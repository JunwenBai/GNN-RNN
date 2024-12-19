import geopandas as gpd
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable

MAP_FILE = "../map/gz_2010_us_050_00_20m/"

def plot_histogram(values, range, column_name, output_dir):
    values = values[~np.isnan(values)]  # Remove NaN
    plt.hist(values, range=range)
    plt.title(column_name)
    plt.savefig(os.path.join(output_dir, "histogram_" + column_name + ".png"))
    plt.close()


# Plot a single feature on a map (used for debugging)
def plot_county_data(counties, values, column_name, year, output_dir):
    
    # Create data frame mapping county FIPS to attribute value
    data_table = pd.DataFrame({'fips': counties, column_name: values})
    # print("Plotting county data")
    # print(data_table.head())

    # Read county map from file, and compute county FIPS
    county_map = gpd.read_file(MAP_FILE)
    county_map['fips'] = county_map['STATE'].astype(int) * 1000 + county_map['COUNTY'].astype(int)
    # print(county_map.head())
    assert(len(set(counties)) == len(counties))  # Make sure no duplicate county entries

    # Add the variable values to the county map
    county_map = county_map.merge(data_table, on='fips')
    fig, ax = plt.subplots(1, 1, figsize=(45, 30))
    county_map.plot(column=column_name, ax=ax, legend=True,
                    missing_kwds={  # Red hatch for missing values. See https://geopandas.org/docs/user_guide/mapping.html#choropleth-maps
                        "color": "lightgrey",
                        # "edgecolor": "red",
                        # "hatch": "///",
                        "label": "Missing values",
                    })
    ax.set_title(column_name + ": " + str(year), fontdict={'fontsize': 30})
    fig.axes[0].tick_params(labelsize=24)
    fig.axes[1].tick_params(labelsize=24)
    plt.savefig(os.path.join(output_dir, column_name + "_" + str(year) + ".png"))
    plt.close()


# Plot true vs. predicted yield on a scatter plot and map. "results_df" is assumed to have a
# column "fips" with the county ID, and there should be at most one row per county.
# For each crop listed in "crop_types", results_df should have columns named 
# "true_{crop}" and "predicted_{crop},"
def plot_true_vs_predicted(results_df, crop_types, description, output_dir):

    # Read county map from file, and compute county FIPS
    county_map = gpd.read_file(MAP_FILE)
    county_map = county_map[~county_map['STATE'].astype(int).isin([2,15,72,11])]  # Remove Alaska, Hawaii, DC, Puerto Rico
    county_map['fips'] = county_map['STATE'].astype(int) * 1000 + county_map['COUNTY'].astype(int)
    assert(len(set(results_df['fips'])) == len(results_df['fips']))  # Make sure no duplicate county entries

    # Add the variable values to the county map
    county_map = county_map.merge(results_df, on='fips', how='left')
    # print(county_map.head())

    # Plot maps of true and predicted yield
    for crop_type in crop_types:
        # Compute difference (predicted - true)
        county_map.loc[county_map["true_" + crop_type].isnull(), "predicted_" + crop_type] = np.nan  # If true value is NaN, make prediction NaN so that maps look similar
        county_map["difference_" + crop_type] = county_map["predicted_" + crop_type] - county_map["true_" + crop_type]

        # Set up subplots
        fig, axeslist = plt.subplots(3, 1, figsize=(30, 38))

        # Compute range of colorbars (by taking the min across both true and predicted)
        min_yield = min(county_map["predicted_" + crop_type].min(), county_map["true_" + crop_type].min())
        max_yield = max(county_map["predicted_" + crop_type].max(), county_map["true_" + crop_type].max())
        max_absolute_difference =  county_map["difference_" + crop_type].abs().max()

        # Customize colorbar
        divider = make_axes_locatable(axeslist[0])
        cax = divider.append_axes("right", size="2%", pad=0)
        cax.tick_params(labelsize=60)  # Modify colorbar font size
        cax.set_ylabel("Yield (bushels/acre)")

        # Plot predicted yields
        county_map.plot(column="predicted_" + crop_type, ax=axeslist[0], legend=True, cmap='viridis',
                        vmin=min_yield, vmax=max_yield, edgecolor="black", cax=cax,
                        missing_kwds={  # Red hatch for missing values. See https://geopandas.org/docs/user_guide/mapping.html#choropleth-maps
                            "color": "lightgrey",
                            # "edgecolor": "red",
                            # "hatch": "///",
                            "label": "Missing values",
                        })

        # True yields
        divider = make_axes_locatable(axeslist[1])
        cax = divider.append_axes("right", size="2%", pad=0)
        cax.tick_params(labelsize=60)
        cax.set_ylabel("Yield (bushels/acre)")
        county_map.plot(column="true_" + crop_type, ax=axeslist[1], legend=True, cmap='viridis',
                        vmin=min_yield, vmax=max_yield, edgecolor="black", cax=cax,
                        missing_kwds={  # Red hatch for missing values. See https://geopandas.org/docs/user_guide/mapping.html#choropleth-maps
                            "color": "lightgrey",
                            # "edgecolor": "red",
                            # "hatch": "///",
                            "label": "Missing values",
                        })

        # Difference (Predicted - True)
        divider = make_axes_locatable(axeslist[2])
        cax = divider.append_axes("right", size="2%", pad=0)
        cax.tick_params(labelsize=60)
        cax.set_ylabel("Yield (bushels/acre)")

        county_map.plot(column="difference_" + crop_type, ax=axeslist[2], legend=True, cmap='RdYlBu',
                        vmin=-max_absolute_difference, vmax=max_absolute_difference, edgecolor="black", cax=cax,
                        missing_kwds={  # Red hatch for missing values. See https://geopandas.org/docs/user_guide/mapping.html#choropleth-maps
                            "color": "lightgrey",
                            # "edgecolor": "red",
                            # "hatch": "///",
                            "label": "Missing values",
                        })        

        axeslist[0].set_title("Predicted yield (bu/ac): " + description + ", " + crop_type, fontsize=60)
        axeslist[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # axeslist[0].tick_params(labelsize=36)
        axeslist[1].set_title("True yield (bu/ac): " + description + ", " + crop_type, fontsize=60)
        axeslist[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # axeslist[1].tick_params(labelsize=24)
        axeslist[2].set_title("Difference (Predicted - True)", fontsize=60)
        axeslist[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # axeslist[2].tick_params(labelsize=24)
        plt.savefig(os.path.join(output_dir, "true_vs_predicted_map_" + crop_type + "_" + description + ".png"), bbox_inches='tight')
        plt.close()
    
    # Create scatter plots: true on x-axis, predicted on y-axis
    rows = math.ceil(len(crop_types) / 2)
    cols = min(len(crop_types), 2)
    fig, axeslist = plt.subplots(rows, cols, figsize=(7*cols, 7*rows), squeeze=False)  # squeeze=False means that even if we only have one plot, axeslist will still be a 2D array
    if len(crop_types) != 1:
        fig.suptitle('Predicted vs true yield: ' + description, fontsize=22)

    for idx, crop_type in enumerate(crop_types):
        ax = axeslist.ravel()[idx]

        predicted = results_df["predicted_" + crop_type].to_numpy()
        true = results_df["true_" + crop_type].to_numpy()

        # Remove rows where we don't have true label
        not_nan = ~np.isnan(true)
        predicted = predicted[not_nan]
        true = true[not_nan]

        # Fit linear regression
        true = true.reshape(-1, 1)
        true_to_predicted = LinearRegression(fit_intercept=False).fit(true, predicted)
        slope = true_to_predicted.coef_[0]
        regression_line = slope * true
        regression_equation = 'y={:.2f}x'.format(slope)
        identity_line = true

        # Compute statistics
        predicted = predicted.ravel()
        true = true.ravel()
        r2 = r2_score(true, predicted)
        corr = np.corrcoef(true, predicted)[0, 1]
        rmse = math.sqrt(mean_squared_error(true, predicted)) 

        # Plot
        ax.scatter(true, predicted, color="k", s=5)
        ax.plot(true, regression_line, 'r', label=regression_equation + ' (R^2={:.2f})'.format(r2))
        ax.plot(true, identity_line, 'g--', label='Identity function')
        ax.tick_params(labelsize=22)
        ax.set_xlabel("True yield (bu/ac)", fontsize=22)
        ax.set_ylabel("Predicted yield (bu/ac)", fontsize=22)
        if len(crop_types) != 1:
            ax.set_title(crop_type, fontsize=22)
        else:
            ax.set_title('Predicted vs true yield: ' + description + ", " + crop_type, fontsize=22)
        # ax.set(xlabel='True yield (bushels/acre)', ylabel='Predicted yield (bushels/acre)', title=crop_type)
        ax.legend(fontsize=22)
        # ax.set_title(crop_type)
    plt.tight_layout()
    # fig.subplots_adjust(top=0.93)
    if len(crop_types) == 1:
        description = crop_types[0] + "_" + description
    plt.savefig(os.path.join(output_dir, "true_vs_predicted_scatter_" + description + ".png"), bbox_inches='tight')
    plt.close()


def sanity_check_input(X, counties, year, args, X_mean, X_std, output_dir="exploratory"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_w = args.time_intervals*args.num_weather_vars 
    print("X shape", X.shape)
    X = (X * X_std) + X_mean

    weather_indices = [0, 2, 3, 4]  # ppt, tmax, tmean, tmin
    weather_vars = ["ppt", "tmax", "tmean", "tmin"]
    for i in range(0, 3):  # X.shape[0]:
        for j in range(0, 5):
            X_w = X[i, j, :n_w].reshape(args.num_weather_vars, args.time_intervals)      
            current_county = counties[i].item()
            current_year = year - 4 + j
            fig, axeslist = plt.subplots(2, 2, figsize=(10, 10), squeeze=False)
            fig.suptitle('Weather vars: county ' + str(current_county) + ', year ' + str(current_year))
            weeks = list(range(1, 53))
            for k in range(len(weather_indices)):
                ax = axeslist.ravel()[k]
                ax.plot(weeks, X_w[weather_indices[k], :])
                ax.set(xlabel='Week', ylabel=weather_vars[k], title=weather_vars[k])
            plt.tight_layout()
            fig.subplots_adjust(top=0.90)
            plt.savefig(os.path.join(output_dir, "weather_vars_county_" + str(current_county) + "_year_" + str(current_year) + ".png"))
            plt.close()           


