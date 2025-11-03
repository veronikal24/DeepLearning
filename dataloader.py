import os
import sys
import random
import pyarrow
import numpy as np
import pandas as pd
import pyarrow.parquet
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from pyproj import Transformer

THIS_PATH = os.path.dirname(os.path.realpath(__file__))
# MY_FILE = THIS_PATH + "\\aisdk-2025-02-27"
MY_FILE = "aisdk-2025-10-20"

# Transformer for lat/lon -> meters (Web Mercator)
_transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)


def csv_to_parquet(file_path, out_path):
    dtypes = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
    }
    usecols = list(dtypes.keys())
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtypes)

    # Remove errors
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[
        (df["Latitude"] <= north)
        & (df["Latitude"] >= south)
        & (df["Longitude"] >= west)
        & (df["Longitude"] <= east)
    ]

    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(
        columns=["Type of mobile"]
    )
    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )

    df = df.drop_duplicates(
        [
            "Timestamp",
            "MMSI",
        ],
        keep="first",
    )

    def track_filter(g):
        len_filt = len(g) > 256  # Min required length of track/segment
        sog_filt = 1 <= g["SOG"].max() <= 50  # Remove stationary tracks/segments
        time_filt = (
            g["Timestamp"].max() - g["Timestamp"].min()
        ).total_seconds() >= 60 * 60  # Min required timespan
        return len_filt and sog_filt and time_filt

    # Track filtering
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(["MMSI", "Timestamp"])

    # Divide track into segments based on timegap
    df["Segment"] = df.groupby("MMSI")["Timestamp"].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum()
    )  # Max allowed timegap

    # Segment filtering
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df = df.reset_index(drop=True)

    #
    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]

    # Clustering
    # kmeans = KMeans(n_clusters=48, random_state=0)
    # kmeans.fit(df[["Latitude", "Longitude"]])
    # df["Geocell"] = kmeans.labels_
    # centers = kmeans.cluster_centers_
    # "Latitude": center[0],
    # "Longitude": center[1],

    # df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")
    # Save as parquet file with partitions
    # pyarrow.Table.from_pandas is the correct constructor (from_pd does not exist)
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=[
            "MMSI",  # "Date",
            "Segment",  # "Geocell"
        ],
    )


def load_parquet(parquet_dir, k=5, seed=42):
    # List all MMSI directories
    mmsi_dirs = sorted(
        d
        for d in os.listdir(parquet_dir)
        if os.path.isdir(os.path.join(parquet_dir, d))
    )
    if len(mmsi_dirs) == 0:
        raise ValueError("No MMSI partitions found.")

    # Sample k MMSIs
    random.seed(seed)
    sample_mmsis = random.sample(mmsi_dirs, min(k, len(mmsi_dirs)))

    # Load all Parquet files for selected MMSIs
    dfs = []
    for mmsi in sample_mmsis:
        mmsi_path = os.path.join(parquet_dir, mmsi)
        try:
            df = pyarrow.parquet.ParquetDataset(mmsi_path).read_pandas().to_pandas()
            df["MMSI"] = mmsi
            dfs.append(df)
        except:
            print("Parquet file corrupted or smth: " + mmsi)

    # Combine
    df = pd.concat(dfs, ignore_index=True)

    # Project lat/lon to meters for fast distance computations
    df["x"], df["y"] = _transformer.transform(
        df["Longitude"].values, df["Latitude"].values
    )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)

    return df


def preprocess_data(df, max_speed_kmh=100, max_time=4, max_dist=50):
    df = df.copy()
    df = df.sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)

    df["x_prev"] = df.groupby("MMSI")["x"].shift()
    df["y_prev"] = df.groupby("MMSI")["y"].shift()
    df["t_prev"] = df.groupby("MMSI")["Timestamp"].shift()

    # Vectorized distance in km
    dx = df["x"] - df["x_prev"]
    dy = df["y"] - df["y_prev"]
    df["dist_km"] = np.sqrt(dx**2 + dy**2) / 1000  # meters -> km

    # Time difference in hours
    df["dt_h"] = (df["Timestamp"] - df["t_prev"]).dt.total_seconds() / 3600

    # Speed in km/h
    df["speed_kmh"] = df["dist_km"] / df["dt_h"].replace(0, np.nan)

    # Keep points under speed limit or first point of trajectory, as long as timestep has no 4h delta or dist has no 50km delta
    df = df[
        ((df["speed_kmh"] <= max_speed_kmh) | (df["speed_kmh"].isna()))
        & (df["dt_h"] <= max_time)
        & (df["dist_km"] <= max_dist)
    ].reset_index(drop=True)

    # Drop temporary columns
    df.drop(
        columns=["x_prev", "y_prev", "t_prev", "dist_km", "dt_h", "speed_kmh"],
        inplace=True,
    )
    return df


def plot_paths_on_map(df, heat=0):
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


def get_ID_by_coords(df, lat, long):
    x_query, y_query = _transformer.transform(long, lat)
    # Vectorized Euclidean distance
    dx = df["x"].values - x_query
    dy = df["y"].values - y_query
    distances = np.sqrt(dx**2 + dy**2)

    # Find index of minimum distance
    idx_min = np.argmin(distances)

    closest_mmsi = df.iloc[idx_min]["MMSI"]

    return closest_mmsi


if __name__ == "__main__":
    csv_to_parquet(
        os.path.join(sys.argv[1]),
        os.path.join("dataset"),
    )
    # df = load_parquet(os.path.join(THIS_PATH, MY_FILE), k=10000)
    # print(df.head())
    # df = preprocess_data(df)
    # plot_paths_on_map(df, heat=[(9, 0.01), (6, 0.02), (3, 0.1)])
    # weird = get_ID_by_coords(df, 54.16, 9.50)
    # plot_paths_on_map(df[df["MMSI"] == weird])
