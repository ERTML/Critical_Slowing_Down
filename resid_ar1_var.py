import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import rasterio
import statsmodels.api as sm
from tqdm import tqdm

### Code for Trend Estimation ###
def calc_ar1(x):
    """计算序列的AR(1)系数"""
    try:
        date_X1 = x[:-1]
        date_X = sm.add_constant(date_X1)
        date_Y = x[1:]
        results = sm.OLS(date_Y, date_X, missing="drop").fit()
        return results.params[1]
    except Exception as e:
        print("error calc ar1", e)
        return np.nan


def sliding_window_calc(x, win_size):
    """滑动窗口计算AR1和方差"""
    try:
        var = np.full_like(x, np.nan, dtype=np.float32)
        ar1 = np.full_like(x, np.nan, dtype=np.float32)
        half_window = int(win_size / 2)
        ln = x.shape[0]
        for i in range(half_window, ln - half_window):
            subset = x[i - half_window: i + half_window]
            try:
                ar1[i] = calc_ar1(subset)
                var[i] = np.nanvar(subset, ddof=1)
            except Exception:
                ar1[i] = np.nan
                var[i] = np.nan
        return ar1, var
    except Exception as e:
        print("error sliding window calc", e)


def process_row(args):
    """子进程：计算单行所有像元"""
    row_idx, row_data, win_size = args
    bands, width = row_data.shape
    ar_row = np.full_like(row_data, np.nan, dtype=np.float32)
    var_row = np.full_like(row_data, np.nan, dtype=np.float32)

    for j in range(width):
        ts = row_data[:, j]
        if np.isnan(ts).all():
            continue
        ar, var = sliding_window_calc(ts, win_size)
        ar_row[:, j] = ar
        var_row[:, j] = var

    return row_idx, ar_row, var_row


def compute_resid_ar1_var(resid_3d, win_size, max_workers=50):
    """
    Core computation: sliding-window AR(1) and variance for 3D residual array.

    Parameters
    ----------
    resid_3d : ndarray (T, H, W)
        Residual time series
    win_size : int
        Sliding window size
    max_workers : int

    Returns
    -------
    ar_all : ndarray (T, H, W)
    var_all : ndarray (T, H, W)
    """
    T, H, W = resid_3d.shape

    ar_all = np.full_like(resid_3d, np.nan, dtype=np.float32)
    var_all = np.full_like(resid_3d, np.nan, dtype=np.float32)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_row, (i, resid_3d[:, i, :], win_size))
            for i in range(H)
        ]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Computing AR1 & VAR", ncols=90 ):
            row_idx, ar_row, var_row = f.result()
            ar_all[:, row_idx, :] = ar_row
            var_all[:, row_idx, :] = var_row

    return ar_all, var_all

def trim_valid_time(ar_all, var_all, win_size):
    half_window = win_size // 2
    valid_idx = slice(half_window, -half_window)
    return ar_all[valid_idx], var_all[valid_idx]


def save_ar_var_rasters(ar_data, var_data, profile, ar_path, var_path):
    profile = profile.copy()
    profile.update(count=ar_data.shape[0])

    with rasterio.open(ar_path, "w", **profile) as dst:
        dst.write(ar_data)
    print(f"AR(1) saved: {ar_path}")

    with rasterio.open(var_path, "w", **profile) as dst:
        dst.write(var_data)
    print(f"Variance saved: {var_path}")


def resid_ar1_var_pipeline(
    resid_3d,
    profile,
    output_ar_raster,
    output_var_raster,
    win_size,
    max_workers=50
):
    start_time = time.time()

    ar_all, var_all = compute_resid_ar1_var(
        resid_3d,
        win_size=win_size,
        max_workers=max_workers
    )

    ar_valid, var_valid = trim_valid_time(ar_all, var_all, win_size)

    save_ar_var_rasters(
        ar_valid,
        var_valid,
        profile,
        output_ar_raster,
        output_var_raster
    )

    print(
        f"Finished win_size={win_size}, "
        f"time={(time.time() - start_time)/60:.1f} min"
    )


def read_raster_3d(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
    return data, profile


def run_multi_window_pipeline(
    resid_3d,
    profile,
    out_dir,
    base_name,
    win_sizes,
    max_workers
):
    os.makedirs(out_dir, exist_ok=True)

    for win_size in win_sizes:
        print(f"\n=== Window size: {win_size} ===")

        ar_path = os.path.join(out_dir, f"{base_name}_AR1_{win_size}.tif")
        var_path = os.path.join(out_dir, f"{base_name}_VAR_{win_size}.tif")

        resid_ar1_var_pipeline(
            resid_3d,
            profile,
            ar_path,
            var_path,
            win_size,
            max_workers
        )


def main(
    input_raster,
    out_dir,
    win_sizes,
    max_workers
):
    print("Reading residual raster...")
    resid_3d, profile = read_raster_3d(input_raster)

    base_name = os.path.splitext(os.path.basename(input_raster))[0]

    run_multi_window_pipeline(
        resid_3d=resid_3d,
        profile=profile,
        out_dir=out_dir,
        base_name=base_name,
        win_sizes=win_sizes,
        max_workers=max_workers
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sliding-window AR(1) and variance computation for residual raster time series"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input residual GeoTIFF (T, H, W)"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--win_sizes",
        type=int,
        nargs="+",
        required=True,
        help="Sliding window sizes, e.g. 36 48 60"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=50,
        help="Number of parallel processes"
    )

    args = parser.parse_args()

    main(
        input_raster=args.input,
        out_dir=args.out_dir,
        win_sizes=args.win_sizes,
        max_workers=args.max_workers
    )