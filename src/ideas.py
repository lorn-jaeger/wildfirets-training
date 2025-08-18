# generic_binary_dataset.py
# Minimal refactor to make arbitrary binary pixel targets easy (PM2.5>100, fire mask, etc.)
# Key changes:
#  - Name bands (schema), stop relying on "last band is fire".
#  - Declare target in a small TargetSpec (source band, threshold/op, lead).
#  - Optional derived channels are declared (e.g., AF binary, landcover one-hot, sinify angles).
#  - Positive-mining crop scores by the declared target y, not by fire.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import glob, warnings
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import h5py
import torchvision.transforms.functional as TF

# ---------- Config primitives ----------

@dataclass
class BandInfo:
    name: str
    kind: str                   # "dynamic" or "static"
    is_degree: bool = False     # apply sin transform if True
    one_hot_classes: int = 0    # >0 means integer-coded landcover to one-hot with this many classes

@dataclass
class BandSchema:
    # order matters: this is the on-disk order of channels in each TIFF/HDF5 frame
    bands: List[BandInfo]

    def index_of(self, name: str) -> int:
        for i, b in enumerate(self.bands):
            if b.name == name:
                return i
        raise KeyError(f"Band '{name}' not found in schema.")

    @property
    def degree_indices(self) -> List[int]:
        return [i for i,b in enumerate(self.bands) if b.is_degree]

@dataclass
class TargetSpec:
    # define a binary pixel target from a band and a threshold
    band: str                  # band name in schema, e.g., "pm25" or "active_fire"
    op: str = ">"              # one of {">", ">=", "==", "<", "<="}
    threshold: float = 100.0   # e.g., 100 for PM2.5
    lead: int = 1              # predict next-day by default (T -> T+lead)
    # optional postprocessing on the raw band before thresholding (e.g., hhmm->hour)
    preproc: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

@dataclass
class DerivedChannel:
    # add extra covariates from existing bands
    name: str
    make: Callable[[torch.Tensor, BandSchema], torch.Tensor]  # takes x(T,C,H,W) BEFORE standardize, returns (T,1,H,W)
    place_after_standardize: bool = False                     # if True, make() sees standardized x

# ---------- Helpful factories for common deriveds ----------

def binary_from_band(band_name: str, predicate: Callable[[torch.Tensor], torch.Tensor], new_name: str):
    def maker(x: torch.Tensor, schema: BandSchema) -> torch.Tensor:
        idx = schema.index_of(band_name)
        return predicate(x[:, idx:idx+1, ...]).float()
    return DerivedChannel(new_name, maker, place_after_standardize=False)

def sinify_degrees(x: torch.Tensor, schema: BandSchema) -> torch.Tensor:
    if len(schema.degree_indices) == 0: return x
    x = x.clone()
    x[:, schema.degree_indices, ...] = torch.sin(torch.deg2rad(x[:, schema.degree_indices, ...]))
    return x

# Example: AF band preproc for hhmm->hour and NaN->0
def af_hhmm_to_hour(t: torch.Tensor) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0)
    return torch.floor_divide(t, 100)

# ---------- The generic dataset ----------

class GenericBinarySegDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        years: List[int],
        schema: BandSchema,
        target: TargetSpec,
        n_obs: int = 1,
        crop_side_length: Optional[int] = 128,
        is_train: bool = True,
        load_from_hdf5: bool = True,
        stats_years: Optional[List[int]] = None,
        center_crop_eval: bool = False,
        derived: Optional[List[DerivedChannel]] = None,
        features_to_keep: Optional[List[str]] = None,   # keep by BAND NAMES
        means: Optional[Dict[str, float]] = None,       # per-band stats by name
        stds: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.years = years
        self.schema = schema
        self.target = target
        self.n_obs = int(n_obs)
        self.crop_side_length = crop_side_length
        self.is_train = is_train
        self.load_from_hdf5 = load_from_hdf5
        self.center_crop_eval = center_crop_eval
        self.derived = derived or []
        self.features_to_keep = features_to_keep
        self.means = means
        self.stds = stds

        # inventory
        self.imgs_per_fire = self._index_files()
        self.skip_initial = 0
        self.datapoints_per_fire = self._compute_datapoints_per_fire()
        self.length = sum([sum(d.values()) for d in self.datapoints_per_fire.values()])

        # stats tensors in channel order (C,)
        if self.means is not None and self.stds is not None:
            m = [self.means[b.name] for b in self.schema.bands]
            s = [self.stds[b.name] for b in self.schema.bands]
            self.means_t = torch.tensor(m)[None, :, None, None]
            self.stds_t = torch.tensor(s)[None, :, None, None]
        else:
            self.means_t, self.stds_t = None, None

        # optional band name filter -> indices
        if self.features_to_keep is not None:
            self.keep_idx = [self.schema.index_of(n) for n in self.features_to_keep]
        else:
            self.keep_idx = None

    def __len__(self): return self.length

    def _index_files(self):
        out = {}
        for y in self.years:
            out[y] = {}
            if self.load_from_hdf5:
                for f in sorted(glob.glob(f"{self.data_dir}/{y}/*.hdf5")):
                    out[y][Path(f).stem] = [f]
            else:
                for d in sorted(glob.glob(f"{self.data_dir}/{y}/*/")):
                    fire = Path(d).name
                    tif_paths = sorted(glob.glob(f"{d}/*.tif"))
                    out[y][fire] = tif_paths
                    if len(tif_paths) == 0:
                        warnings.warn(f"{y}/{fire} has no TIFFs.", RuntimeWarning)
        return out

    def _compute_datapoints_per_fire(self):
        out = {}
        for y, fires in self.imgs_per_fire.items():
            out[y] = {}
            for fire, files in fires.items():
                if self.load_from_hdf5:
                    if not files:
                        n = 0
                    else:
                        with h5py.File(files[0], "r") as f:
                            n = len(f["data"])
                else:
                    n = len(files)
                dp = n - self.n_obs - self.target.lead + 1
                out[y][fire] = max(0, dp)
        return out

    # map global index -> (year, fire, in_fire_index)
    def _locate(self, idx: int) -> Tuple[int, str, int]:
        if idx < 0: idx = self.length + idx
        if idx >= self.length: raise IndexError("dataset index out of range")
        seen = 0
        for y in self.datapoints_per_fire:
            for fire, dp in self.datapoints_per_fire[y].items():
                if idx < seen + dp:
                    return y, fire, idx - seen
                seen += dp
        raise RuntimeError("indexing bug")

    def _load_window(self, y: int, fire: str, k: int):
        # window = [k ... k+n_obs-1] for x, and target day = k+n_obs-1+lead
        start = k
        x_end = k + self.n_obs
        y_day = x_end - 1 + self.target.lead

        if self.load_from_hdf5:
            fpath = self.imgs_per_fire[y][fire][0]
            with h5py.File(fpath, "r") as f:
                arr_x = f["data"][start:x_end]           # (T,C,H,W)
                arr_y = f["data"][y_day]                 # (C,H,W)
        else:
            all_paths = self.imgs_per_fire[y][fire]
            stack = []
            for p in all_paths[start:x_end]:
                with rasterio.open(p) as ds: stack.append(ds.read())
            arr_x = np.stack(stack, axis=0)
            with rasterio.open(all_paths[y_day]) as ds: arr_y = ds.read()

        return arr_x, arr_y

    def __getitem__(self, idx):
        y, fire, k = self._locate(idx)
        x_np, y_np_full = self._load_window(y, fire, k)
        x = torch.tensor(x_np)            # (T,C,H,W)
        y_raw = torch.tensor(y_np_full)   # (C,H,W)

        # band preprocs (example: AF hhmm->hour)
        # apply to x for bands that need it
        # you can add more per-band preprocs if needed
        # ex: if a band is called "active_fire", convert hhmm->hour
        try:
            af_idx = self.schema.index_of("active_fire")
            t = x[:, af_idx, ...]
            x[:, af_idx, ...] = af_hhmm_to_hour(t)
        except KeyError:
            pass

        # target build
        t_idx = self.schema.index_of(self.target.band)
        t_slice = y_raw[t_idx, ...]                        # (H,W)
        t_t = torch.nan_to_num(t_slice, nan=0.0).float()
        if self.target.preproc is not None:
            t_t = self.target.preproc(t_t)
        if self.target.op == ">":   y_bin = (t_t >  self.target.threshold)
        elif self.target.op == ">=":y_bin = (t_t >= self.target.threshold)
        elif self.target.op == "==":y_bin = (t_t == self.target.threshold)
        elif self.target.op == "<": y_bin = (t_t <  self.target.threshold)
        elif self.target.op == "<=":y_bin = (t_t <= self.target.threshold)
        else: raise ValueError(f"Unsupported op {self.target.op}")
        y = y_bin.long()                                     # (H,W)

        # training/eval spatial transforms
        if self.is_train and self.crop_side_length:
            x, y = self._pos_mining_crop(x, y, self.crop_side_length)
        elif (not self.is_train) and self.center_crop_eval and self.crop_side_length:
            x, y = self._center_crop(x, y, self.crop_side_length)

        # angle -> sin
        if len(self.schema.degree_indices) > 0:
            x[:, self.schema.degree_indices, ...] = torch.sin(torch.deg2rad(x[:, self.schema.degree_indices, ...]))

        # feature selection by names, if requested
        if self.keep_idx is not None:
            x = x[:, self.keep_idx, ...]

        # standardize if stats provided
        if self.means_t is not None and self.stds_t is not None:
            # broadcast to (1,C,1,1) across time T later
            xm = self.means_t
            xs = self.stds_t
            x = (x - xm) / (xs + 1e-8)

        # derived channels (pre-std ones ran above; post-std set place_after_standardize=True)
        for d in self.derived:
            ch = d.make(x, self.schema) if d.place_after_standardize else d.make(x, self.schema)
            x = torch.cat([x, ch], dim=1)

        # landcover one-hot if present
        for i, b in enumerate(self.schema.bands):
            if b.one_hot_classes > 0:
                lc = x[:, i, ...].long().clamp(min=1) - 1
                oh = torch.eye(b.one_hot_classes)[lc.flatten()].reshape(x.shape[0], x.shape[2], x.shape[3], b.one_hot_classes).permute(0,3,1,2)
                # replace the LC integer channel with the one-hot block
                x = torch.cat([x[:, :i, ...], oh, x[:, i+1:, ...]], dim=1)
                break

        x = torch.nan_to_num(x, nan=0.0)

        return (x, y)

    # --------- crops ---------
    def _center_crop(self, x, y, S):
        return TF.center_crop(x, (S, S)), TF.center_crop(y, (S, S))

    def _pos_mining_crop(self, x, y, S):
        T, C, H, W = x.shape
        if (S is None) or (S > H) or (S > W):
            return x, y
        best_score = -1
        best = (None, None)
        for _ in range(10):
            top = np.random.randint(0, H - S + 1)
            left = np.random.randint(0, W - S + 1)
            xc = TF.crop(x, top, left, S, S)
            yc = TF.crop(y, top, left, S, S)
            score = yc.float().mean()  # focus on positive target density; customize if you want
            if score > best_score:
                best_score, best = score, (xc, yc)
        x, y = best
        # random flips/rotations with degree-fix not needed here because we sinified degrees already
        if np.random.rand() > 0.5:
            x = TF.hflip(x); y = TF.hflip(y)
        if np.random.rand() > 0.5:
            x = TF.vflip(x); y = TF.vflip(y)
        rot = np.random.randint(0, 4)
        if rot:
            angle = int(rot * 90)
            x = TF.rotate(x, angle)
            y = TF.rotate(y.unsqueeze(0), angle).squeeze(0)
        return x, y


