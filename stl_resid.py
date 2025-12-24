import os
import numpy as np
import rasterio
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
from joblib import Parallel, delayed
from contextlib import contextmanager


@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Enable tqdm progress bar for joblib.Parallel
    """
    from joblib import parallel
    original_callback = parallel.BatchCompletionCallBack

    class TqdmCallback(original_callback):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return original_callback.__call__(self, *args, **kwargs)

    parallel.BatchCompletionCallBack = TqdmCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = original_callback
        tqdm_object.close()


def robust_stl(series, period, smooth_length=7):
    ### REFERENCE FOR LOESS PARAMS BASED ON ORIGINAL FORTRAN CODE ###
    # np = f              # period of seasonal component
    # ns = 7              # length of seasonal smoother
    # nt = nt_calc(f,ns)  # length of trend smoother
    # nl = nl_calc(f)     # length of low-pass filter
    # isdeg = 1           # Degree of locally-fitted polynomial in seasonal smoothing.
    # itdeg = 1           # Degree of locally-fitted polynomial in trend smoothing.
    # ildeg = 1           # Degree of locally-fitted polynomial in low-pass smoothing.
    # nsjump = None       # Skipping value for seasonal smoothing.
    # ntjump = 1          # Skipping value for trend smoothing. If None, ntjump= 0.1*nt
    # nljump = 1          # Skipping value for low-pass smoothing. If None, nljump= 0.1*nl
    # robust = True       # Flag indicating whether robust fitting should be performed.
    # ni = 1              # Number of loops for updating the seasonal and trend  components.
    # no = 3              # Number of iterations of robust fitting. The value of no should
    #                       be a nonnegative integer. If the data are well behaved without
    #                       outliers, then robustness iterations are not needed. In this case
    #                       set no=0, and set ni=2-5 depending on how much security you want
    #                       that the seasonal-trend looping converges. If outliers are present
    #                       then no=3 is a very secure value unless the outliers are radical,
    #                       in which case no=5 or even 10 might be better. If no>0 then set ni
    #                       to 1 or 2. If None, then no is set to 15 for robust fitting,
    #                       to 0 otherwise.
    def nt_calc(f, ns):
        '''Calcualte the length of the trend smoother based on Cleveland et al., 1990'''
        nt = int((1.5 * f) / (1 - 1.5 * (1 / ns)) + 1) # Force fractions to be rounded up
        if nt % 2 == 0:
            nt += 1
        return nt
    def nl_calc(f):
        '''Calcualte the length of the low-pass filter based on Cleveland et al., 1990'''
        nl = int(f)
        if nl % 2 == 0:
            nl += 1
        if nl <= f:
            nl += 2
        return nl
    try:
        res = STL(series, period, seasonal=smooth_length, trend=nt_calc(period, smooth_length),
                  low_pass=nl_calc(period), seasonal_deg=1, trend_deg=1, low_pass_deg=1,
                  seasonal_jump=1, trend_jump=1, low_pass_jump=1, robust=True)
        return res.fit()
    except Exception as e:
        print(" robust_stl failed ", e)


def stl_resid(arr, period, n_jobs=-1):
    """
    Compute STL residuals for a 3D array (T, H, W)
    Parallelized by rows.
    """
    T, H, W = arr.shape
    print(f"Data shape: T={T}, H={H}, W={W}")
    print(f"Row-wise parallel processing, total rows: {H}")

    def process_series(series):
        """
        STL decomposition for a single pixel time series.
        Only residual component is returned.
        """
        try:
            res = robust_stl(series, period)
            trend = res.trend
            seasonal = res.seasonal
            resid = res.resid
            return resid.astype(np.float32)
        except Exception as e:
            print(" main_stl failed ", e)
            return np.full_like(series, np.nan, dtype=np.float32)

    def process_row(i):
        row = arr[:, i, :]  # (T, W)
        row_out = np.zeros_like(row, dtype=np.float32)
        for j in range(W):
            row_out[:, j] = process_series(row[:, j])
        return i, row_out

    with tqdm_joblib(tqdm(total=H, desc="Row processing")):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_row)(i) for i in range(H)
        )

    resid3d = np.zeros_like(arr, dtype=np.float32)
    for idx, row_data in results:
        resid3d[:, idx, :] = row_data

    return resid3d


def prepare_data(in_dir):
    """
    Read all GeoTIFF files in a directory and stack them into a 3D array
    """
    tif_list = sorted(
        [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(".tif")]
    )
    if len(tif_list) == 0:
        raise ValueError("No .tif files found")

    data_list, profile = [], None

    # Read all temporal slices
    for tif in tif_list:
        with rasterio.open(tif) as src:
            if profile is None:
                profile = src.profile.copy()
            data_list.append(src.read(1))

    arr = np.stack(data_list, axis=0)  # (T, H, W)
    return arr, profile


def save_result(out_path, data, profile):
    """
    Save a 3D array to a multi-band GeoTIFF
    """
    T = data.shape[0]
    profile = profile.copy()
    profile.update(
        dtype=rasterio.float32,
        compress='lzw',
        count=T
    )

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data)


def main(in_dir, out_path, period, n_jobs):
    """
    Main workflow for STL residual extraction.

    Parameters
    ----------
    in_dir : str
        Directory containing time-series GeoTIFF files
    out_path : str
        Output GeoTIFF file path (residual component)
    n_jobs : int
        Number of parallel jobs
    period : int
        Seasonal period (12 for monthly data)
    """
    print(f"STL seasonal period = {period}")

    arr, profile = prepare_data(in_dir)

    # Basic sanity check for monthly data
    if arr.shape[0] < 2 * period:
        raise ValueError(
            f"Time series too short for STL: T={arr.shape[0]}, "
            f"period={period}"
        )

    resid = stl_resid(arr, period, n_jobs=n_jobs)
    save_result(out_path, resid, profile)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="STL residual extraction for kNDVI raster time series"
    )

    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Directory containing monthly kNDVI GeoTIFF files"
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output GeoTIFF path for STL residual"
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (default: all cores)"
    )

    parser.add_argument(
        "--period",
        type=int,
        default=12,
        help="Seasonal period (default: 12 for monthly data)"
    )

    args = parser.parse_args()

    main(
        in_dir=args.in_dir,
        out_path=args.out_path,
        period=args.period,
        n_jobs=args.n_jobs
    )
