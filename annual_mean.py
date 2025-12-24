import numpy as np
import rasterio
from tqdm import tqdm
import os

def generate_month_list(start_year, start_month, band_count):
    ym_list = []
    year, month = start_year, start_month
    for _ in range(band_count):
        ym_list.append(f"{year}{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return ym_list


def compute_valid_years(band_count, win_size, start_year):
    half_win = win_size // 2
    skip_years = int(np.ceil(half_win / 12))
    start_year = start_year + skip_years

    start_idx = half_win % 12
    valid_band_count = band_count - 2 * start_idx
    year_count = valid_band_count // 12

    years = [start_year + i for i in range(year_count)]
    return years, start_idx


def monthly_to_annual_mean(data_3d, start_idx, years):
    annual_list = []

    for i, year in enumerate(tqdm(years, desc=f"Annual mean", ncols=90)):
        idx_start = start_idx + i * 12
        idx_end = idx_start + 12
        subset = data_3d[idx_start:idx_end]
        annual_list.append(np.nanmean(subset, axis=0))

    annual_stack = np.stack(annual_list, axis=0)
    return annual_stack

def read_raster_3d(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        meta = src.meta.copy()
    return data, meta

def write_annual_raster(path, annual_stack, meta, years, prefix):
    meta = meta.copy()
    meta.update(count=len(years))

    with rasterio.open(path, "w", **meta) as dst:
        dst.write(annual_stack)
        dst.descriptions = tuple(f"{prefix}_{y}" for y in years)

def annual_mean_pipeline(input_file, output_file, win_size, start_year, prefix):
    data, meta = read_raster_3d(input_file)

    years, start_idx = compute_valid_years(
        band_count=data.shape[0],
        win_size=win_size,
        start_year=start_year
    )

    print(f"Detected year range: {years[0]}–{years[-1]} ({len(years)} years)")

    annual_stack = monthly_to_annual_mean(
        data,
        start_idx=start_idx,
        years=years
    )

    write_annual_raster(output_file, annual_stack, meta, years, prefix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute annual mean composites from monthly raster time series"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input monthly GeoTIFF (T, H, W)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output annual-mean GeoTIFF"
    )

    parser.add_argument(
        "--start_year",
        type=int,
        required=True,
        help="Start year of the time series (e.g., 1982)"
    )

    parser.add_argument(
        "--win_size",
        type=int,
        default=60,
        help="Sliding window size used (months), default: 60"
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="AR1",
        help="Band name prefix (e.g., AR1, VAR)"
    )

    args = parser.parse_args()

    annual_mean_pipeline(
        input_file=args.input,
        output_file=args.output,
        win_size=args.win_size,
        start_year=args.start_year,
        prefix=args.prefix
    )

