## 概述

本项目提供用于处理时间序列栅格数据（如 NDVI、kNDVI）的 Python 脚本，可实现季节-趋势残差提取、时间自相关和方差计算。

---

## 1. STL 残差提取 (`stl_resid.py`)

**功能：**
对每个像元的时间序列进行稳健的季节-趋势分解（STL），提取残差分量。
* 读取一个目录下的月尺度 GeoTIFF 文件，并堆叠为 3D 数组 `(T, H, W)`。
* 将残差保存为多波段 GeoTIFF。

**使用示例：**

```bash
python stl_resid.py --in_dir /path/to/monthly_knmdvi_tifs \
                    --out_path /path/to/resid.tif \
                    --period 12 \
                    --n_jobs -1
```

---

## 2. AR(1) 与方差计算 (`resid_ar1_var.py`)

**功能：**
使用滑动窗口方法计算残差时间序列的自回归系数 AR(1) 和方差。
* 支持多种滑动窗口尺寸。
* 将 AR(1) 和方差结果保存为多波段 GeoTIFF。

**使用示例：**

```bash
python resid_ar1_var.py --input resid.tif \
                        --out_dir /path/to/output \
                        --win_sizes 36 48 60 \
                        --max_workers 50
```

---

## 3. 年度平均生成 (`annual_mean.py`)

**功能：**
将月度时间序列栅格数据转换为年度平均影像，同时考虑滑动窗口对时间边缘的影响。

**主要特点：**

* 根据滑动窗口大小和起始年份计算有效年份范围。
* 对每个像元计算年度平均值。
* 输出为多波段 GeoTIFF，并为每个波段添加描述信息。

**使用示例：**

```bash
python annual_mean.py --input AR1_resid_60.tif \
                      --output AR1_annual.tif \
                      --start_year 1982 \
                      --win_size 60 \
                      --prefix AR1
```

---

## 工作流程概览

1. **STL 分解**：从月度时间序列中提取残差。
2. **滑动窗口计算**：计算残差的 AR(1) 和方差。
3. **年度平均**：将月度滑动窗口结果生成年度平均值。

---

## 依赖库

* Python ≥ 3.8
* `numpy`
* `rasterio`
* `statsmodels`
* `tqdm`
* `joblib`
