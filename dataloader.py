import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

THIS_PATH = os.path.dirname(os.path.realpath(__file__))


def load_csv(file_path, nrows=None):
    df = pd.read_csv(
        file_path,
        nrows=nrows,
        usecols=[
            "# Timestamp",
            "Type of mobile",
            "MMSI",
            "Latitude",
            "Longitude",
            "Navigational status",
        ],
        sep=",",
        engine="python",
    )
    df.columns = [col.strip().lstrip("# ").strip() for col in df.columns]
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    return df


def filter_some(df):
    df = df[df["Type of mobile"] == "Class A"]
    df = df[df["Navigational status"] == "Under way using engine"]
    return df


def plot_some(df, MMSI=None):
    if not MMSI:
        ax = plt.axes(projection=ccrs.PlateCarree())

        min_lon, max_lon = 5.0, 15
        min_lat, max_lat = 50, 60
        ax.set_extent([min_lon, max_lon, min_lat, max_lat])

        # Add map features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        for mmsi, group in df.groupby("MMSI"):
            group_sorted = group.sort_values("Timestamp")
            ax.plot(
                group_sorted["Longitude"], group_sorted["Latitude"], label=str(mmsi)
            )
            # ax.scatter(group_sorted["Longitude"], group_sorted["Latitude"], s=3)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Connected Scatter of Vessels by MMSI")
        # ax.legend()
        plt.show()


def get_list_of_unique_ids(file_path):
    df = pd.read_csv(
        file_path, nrows=10_000_000, usecols=["MMSI"], sep=",", engine="python"
    )
    return set(df["MMSI"])


if __name__ == "__main__":
    df = load_csv(os.path.join(THIS_PATH, "aisdk-2025-10-20.csv"))

    df = filter_some(df)

    print(len(df))
    # plot_some(df)
