import random
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature


def plot_paths_on_map(df, heat=None):
    """Plots the vessel paths from the given dataset onto a map.
    The optional heat value can be used to plot all paths in the same color to see where traffic is most active.

    Args:
        df (dataframe): AIS data
        heat (list[tuple[int]], optional): A list containing tuples of the to-be-plotted (linewidth, alpha) for each trajectory (Gives a "fade" effect when used with something like "[(9, 0.005), (6, 0.01), (3, 0.05)]"). Defaults to None.
    """
    ax = plt.axes(projection=ccrs.PlateCarree())

    min_lon, max_lon = 5, 15
    min_lat, max_lat = 53, 60
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    if df is not None:
        for mmsi, group in df.groupby("MMSI"):
            group_sorted = group.sort_values("Timestamp")
            if heat:
                for width, alpha in heat:
                    ax.plot(
                        group_sorted["Longitude"],
                        group_sorted["Latitude"],
                        label=str(mmsi),
                        linewidth=width,
                        color="r",
                        alpha=alpha,
                    )
            else:
                ax.plot(
                    group_sorted["Longitude"], group_sorted["Latitude"], label=str(mmsi)
                )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Connected Scatter of Vessels by MMSI")
    # ax.legend()
    plt.show()


def plot_dataset_sample(dataset):
    # Create 4 subplots
    fig, axs = plt.subplots(
        2, 2, figsize=(14, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axs = axs.flatten()

    sample = random.sample(range(len(dataset)), 4)

    for i, ax in enumerate(axs):
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        for x, y in dataset[sample[i] : sample[i] + 1]:
            xn = (x[:, 2].numpy(), x[:, 3].numpy())  # cols 2 and 3 are latitude / lon
            yn = (
                xn[0][-1] + np.cumsum(y[:, 0].numpy()),
                xn[1][-1] + np.cumsum(y[:, 1].numpy()),
            )  # cols 0 and 1 are delta_lat / delta_lon
            min_lon, max_lon = (
                min(min(xn[1]), min(yn[1])) - 1,
                max(max(xn[1]), max(yn[1])) + 1,
            )
            min_lat, max_lat = (
                min(min(xn[0]), min(yn[0])) - 1,
                max(max(xn[0]), max(yn[0])) + 1,
            )
            ax.set_extent([min_lon, max_lon, min_lat, max_lat])
            # ax.scatter(xn[1], xn[0], color="b", s=10)
            # ax.scatter(yn[1], yn[0], color="g", s=10)
            ax.plot(xn[1], xn[0], color="b", linewidth=4)
            ax.plot(yn[1], yn[0], color="g", linewidth=4)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Connected Scatter of Vessels (Sample {i + 1})")

    plt.tight_layout()
    plt.show()


def plot_testresult_sample(dataset):
    # Create 4 subplots
    fig, axs = plt.subplots(
        4, 2, figsize=(14, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axs = axs.flatten()

    sample = random.sample(range(len(dataset)), 8)

    for i, ax in enumerate(axs):
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        for x, y, p in dataset[sample[i] : sample[i] + 1]:
            xn = (x.squeeze(0).numpy()[:, 2], x.squeeze(0).numpy()[:, 3])
            yn = (y.squeeze(0).numpy()[:, 0], y.squeeze(0).numpy()[:, 1])
            yn = (
                xn[0][-1] + np.cumsum(y.squeeze(0).numpy()[:, 0]),
                xn[1][-1] + np.cumsum(y.squeeze(0).numpy()[:, 1]),
            )
            pn = (
                xn[0][-1] + np.cumsum(p.squeeze(0).numpy()[:, 0]),
                xn[1][-1] + np.cumsum(p.squeeze(0).numpy()[:, 1]),
            )
            min_lon, max_lon = (
                min(min(xn[1]), min(yn[1])) - 1,
                max(max(xn[1]), max(yn[1])) + 1,
            )
            min_lat, max_lat = (
                min(min(xn[0]), min(yn[0])) - 1,
                max(max(xn[0]), max(yn[0])) + 1,
            )
            ax.set_extent([min_lon, max_lon, min_lat, max_lat])
            # ax.scatter(xn[1], xn[0], color="b", s=10)
            # ax.scatter(yn[1], yn[0], color="g", s=10)
            ax.plot(xn[1], xn[0], color="b", linewidth=4)
            ax.plot(yn[1], yn[0], color="g", linewidth=4)
            ax.plot(pn[1], pn[0], color="r", linewidth=4)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Connected Scatter of Vessels (Sample {i + 1})")

    plt.tight_layout()
    plt.show()
