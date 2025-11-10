import os
import sys
import torch
import random
import pyarrow
import pyarrow.parquet

import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

from pyproj import Transformer
from collections import defaultdict
from torch.utils.data import Dataset

# Transformer for lat/lon -> meters (Web Mercator)
_transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)


def csv_to_parquet(file_path, out_path):
    """Reads the csv files from the danish thingy and creates parquet files in the given directory (this is the script Heisenberg provided)

    Args:
        file_path (str): path to the input csv
        out_path (str): path for the output
    """
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
    """Loads a randomized sample of the previously created parquet files from the given directory.

    Args:
        parquet_dir (str): directory of the dataset
        k (int, optional): Number of samples. Defaults to 5.
        seed (int, optional): Randomizer seed. Defaults to 42.

    Raises:
        ValueError: If the Parquet reading fails due to corrupted files (is also handled in here)

    Returns:
        pandas.dataframe: pandas dataframe of the loaded data (downsampled to 6 minute intervals)
    """
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
        except Exception:
            print("Parquet file corrupted or smth: " + mmsi)

    # Combine
    df = pd.concat(dfs, ignore_index=True)

    # Project lat/lon to meters for fast distance computations
    df["x"], df["y"] = _transformer.transform(
        df["Longitude"].values, df["Latitude"].values
    )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)

    # Downsample to 6-minute intervals
    df = (
        df.set_index("Timestamp")
        .groupby("MMSI")
        .resample("6T")
        .mean(numeric_only=True)
        .dropna(subset=["Latitude", "Longitude"])
        .reset_index()
    )
    valid_ids = df.groupby("MMSI")["COG"].transform(lambda x: x.notna().all())
    df = df[valid_ids].reset_index(drop=True)

    return df


def preprocess_data(df, max_speed_kmh=100, max_time=4, max_dist=50):
    """Applies filtering methods to the input data, so that we end up with a clean dataset.

    Currently, it just removes data based on:
        1. If the (calculated) speed from one point to the next would exceed 'max_speed_kmh'
        2. If the time delta between two sequential points is larger than 4 hours
        3. If the distance between two sequential points is larger than 50km

    TODO:
        - Check if filter (2) has to be removed (since boats can 'take a break' in a harbor)
        - Implement other filtering methods if needed

    Args:
        df (dataframe): The dataframe to be filtered
        max_speed_kmh (int, optional): Maximum legal speed between two points
        max_time (int, optional): Maximum legal time between two points
        max_dist (int, optional): Maximum legal distance between two points

    Returns:
        pandas.dataframe: pandas dataframe of the filtered data
    """
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


def get_ID_by_coords(df, lat, long):
    """Retrieves the MMSI ID which is closest to the given latitude and longitude. Can be used to find some specific trajectory which looks weird.

    Args:
        df (dataframe): AIS data
        lat (float): latitude
        long (float): longitude

    Returns:
        str: The MMSI of the closest trajectory
    """
    x_query, y_query = _transformer.transform(long, lat)
    # Vectorized Euclidean distance
    dx = df["x"].values - x_query
    dy = df["y"].values - y_query
    distances = np.sqrt(dx**2 + dy**2)

    # Find index of minimum distance
    idx_min = np.argmin(distances)

    closest_mmsi = df.iloc[idx_min]["MMSI"]

    return closest_mmsi


class SlidingWindowDataset(Dataset):
    """Inherits from pytorch dataset;
    Determines blocks of sequences based on the 'max_diff_per_sequence_minutes',
    then extracts windows out of these blocks, based on the given 'window_size_minutes' and 'pred_size_minutes' parameters.
    The 'stride' value determines how much each new extraction within this window is moving to the side.
    """

    def __init__(
        self,
        df,
        max_diff_per_sequence_minutes=6,
        window_size_minutes=60,
        pred_size_minutes=30,
        stride=6,
    ):
        self.df = df.copy()
        self.windows = []
        self.blocks = defaultdict(list)
        self.window_size_minutes = window_size_minutes
        self.pred_size_minutes = pred_size_minutes
        self.stride = stride

        for mmsi, group in df.groupby("MMSI"):
            group = group.sort_values("Timestamp")
            group["time_diff"] = group["Timestamp"].diff()
            group["block"] = (
                group["time_diff"] > pd.Timedelta(minutes=max_diff_per_sequence_minutes)
            ).cumsum()

            for _, block_df in group.groupby("block"):
                if len(block_df) < 2:
                    continue

                self.blocks[mmsi].append(
                    (block_df["Timestamp"].min(), block_df["Timestamp"].max())
                )

                # Compute relative time in minutes
                block_df = block_df.copy()
                block_df["time_min"] = (
                    block_df["Timestamp"] - block_df["Timestamp"].iloc[0]
                ).dt.total_seconds() / 60.0

                if block_df["time_min"].iloc[-1] < window_size_minutes:
                    continue

                start_time = 0.0
                while True:
                    end_time = start_time + window_size_minutes
                    if end_time > block_df["time_min"].iloc[-1]:
                        break

                    # Use aligned masks
                    window_mask = (block_df["time_min"] >= start_time) & (
                        block_df["time_min"] < end_time
                    )
                    if not window_mask.any():
                        start_time += self.stride
                        continue

                    split_time = start_time + (window_size_minutes - pred_size_minutes)
                    input_mask = (block_df["time_min"] >= start_time) & (
                        block_df["time_min"] < split_time
                    )
                    pred_mask = (block_df["time_min"] >= split_time) & (
                        block_df["time_min"] < end_time
                    )

                    input_df = block_df.loc[
                        input_mask, ["Latitude", "Longitude", "SOG", "COG"]
                    ]
                    pred_df = block_df.loc[pred_mask, ["Latitude", "Longitude"]]

                    if len(input_df) == 0 or len(pred_df) == 0:
                        start_time += self.stride
                        continue

                    x = torch.tensor(input_df.values, dtype=torch.float32)
                    y = torch.tensor(pred_df.values, dtype=torch.float32)
                    self.windows.append((x, y))

                    start_time += self.stride

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


# if __name__ == "__main__":
#     ####################################################################################
#     ### Uncomment this block if you want to parse a csv file into the parquet
#     ### data format (call the python code as "python dataloader.py csv_filename.csv")
#     ####################################################################################
#     # csv_to_parquet(
#     #    sys.argv[1],
#     #    "dataset",
#     # )
#     ####################################################################################

#     ####################################################################################
#     ### Uncomment this block if you want to load a sample of the data in the 'dataset'
#     ### directory, apply the preprocessing filtering and plot the heatmap
#     ####################################################################################
#     # df = load_parquet("dataset", k=100)
#     # df = preprocess_data(df)
#     # plot_paths_on_map(df, heat=[(9, 0.01), (6, 0.02), (3, 0.1)])
#     ####################################################################################

#     ####################################################################################
#     ### Uncomment this block if you want to retrieve and plot a single vessel
#     ### trajectory, f.ex. which is closest to the coordinates (54.16, 9.50)
#     ####################################################################################
#     # df = load_parquet("dataset", k=100)
#     # df = preprocess_data(df)
#     # weird = get_ID_by_coords(df, 54.16, 9.50)
#     # plot_paths_on_map(df[df["MMSI"] == weird])
#     ####################################################################################

#     ####################################################################################
#     ### Uncomment this block if you want to load the data sample into a pytorch Dataset
#     ### TODO: Think about if we should maybe include the whole sampling and dataloading stuff into the Dataset class in case we run out of memory
#     ####################################################################################
#     df = load_parquet("dataset", k=100)
#     df = preprocess_data(df)
#     dataset = SlidingWindowDataset(
#         df,
#         max_diff_per_sequence_minutes=30,
#         window_size_minutes=60,
#         pred_size_minutes=30,
#         stride=12,
#     )
#     x, y = dataset[0]
#     print(x)
#     print(y)
#     ####################################################################################
