"""Width-13 Lake Erie dropout uncertainty diagnostic.

The frozen reported result uses 200 predictive draws, obtained as ten
20-draw chunks with sampling seeds 20260824,...,20260833.  The two archived
checkpoints correspond to the support-aware and centroid-trained fits at seed
13 and 900 optimization steps.  The centroid-trained model is evaluated after
correct target-block aggregation (PostAgg).

Use the archived checkpoints (default):
    python code/erie/dropout_uncertainty.py --mode support
    python code/erie/dropout_uncertainty.py --mode centroid

Refit before drawing:
    python code/erie/dropout_uncertainty.py --mode support --refit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["support", "centroid"], required=True)
    p.add_argument("--seed", type=int, default=13, help="training seed")
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--sampling-seed-base", type=int, default=20260824)
    p.add_argument(
        "--input",
        default=str(REPO / "data/raw/LE_CHL_MODIS_SQ_6e88_c718_3d53.csv"),
    )
    p.add_argument(
        "--checkpoint-dir", default=str(REPO / "checkpoints/uncertainty")
    )
    p.add_argument(
        "--outdir", default=str(REPO / "results/erie/uncertainty")
    )
    p.add_argument("--refit", action="store_true")
    return p.parse_args()


def load_fields(path: str):
    cols = [
        "time (UTC)",
        "latitude (degrees_north)",
        "longitude (degrees_east)",
        "chlorophyll (ug/L)",
    ]
    mod = pd.read_csv(
        path,
        usecols=cols,
        dtype={
            "latitude (degrees_north)": "float32",
            "longitude (degrees_east)": "float32",
            "chlorophyll (ug/L)": "float32",
        },
    )
    ts = np.sort(mod["time (UTC)"].unique())
    times = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    lats = np.sort(mod["latitude (degrees_north)"].unique())
    lons = np.sort(mod["longitude (degrees_east)"].unique())
    T, ny, nx = len(times), len(lats), len(lons)
    A = np.full((T, ny, nx), np.nan, np.float32)
    mp = {s: i for i, s in enumerate(ts)}
    it = mod["time (UTC)"].map(mp).to_numpy(np.int64)
    iy = np.searchsorted(lats, mod["latitude (degrees_north)"].to_numpy())
    ix = np.searchsorted(lons, mod["longitude (degrees_east)"].to_numpy())
    A[it, iy, ix] = mod["chlorophyll (ug/L)"].to_numpy(np.float32)

    cnt = np.isfinite(A).sum(0)
    lab, _ = label(cnt >= 3)
    sizes = np.bincount(lab.ravel())
    water = lab == (np.argmax(sizes[1:]) + 1)
    xg = (2 * (lons - lons.min()) / (lons.max() - lons.min()) - 1).astype(
        np.float32
    )
    yg = (2 * (lats - lats.min()) / (lats.max() - lats.min()) - 1).astype(
        np.float32
    )
    return A, water, xg, yg, T, ny, nx


def make_blocks(A, water, xg, yg, shift: int, bs: int = 13, P: int = 64):
    T, ny, nx = A.shape
    rows = []
    for t in range(T):
        V = A[t]
        for y0 in range(shift, ny, bs):
            y1 = min(y0 + bs, ny)
            if y1 - y0 < 6:
                continue
            for x0 in range(shift, nx, bs):
                x1 = min(x0 + bs, nx)
                if x1 - x0 < 6:
                    continue
                wm = water[y0:y1, x0:x1]
                if wm.sum() < 13:
                    continue
                va = wm & np.isfinite(V[y0:y1, x0:x1])
                if va.sum() / wm.sum() < 0.85:
                    continue
                yy, xx = np.where(va)
                yy += y0
                xx += x0
                target = np.log1p(np.mean(V[yy, xx]))
                take = np.linspace(0, len(yy) - 1, min(P, len(yy))).round().astype(int)
                yy = yy[take]
                xx = xx[take]
                q = len(yy)
                xp = np.zeros(P, np.float32)
                yp = np.zeros(P, np.float32)
                ma = np.zeros(P, np.float32)
                xp[:q] = xg[xx]
                yp[:q] = yg[yy]
                ma[:q] = 1
                rows.append((t, target, xp[:q].mean(), yp[:q].mean(), xp, yp, ma))
    return rows


def pack(rows):
    vals = {
        "t": np.array([r[0] for r in rows], np.int64),
        "y": np.array([r[1] for r in rows], np.float32),
        "cx": np.array([r[2] for r in rows], np.float32),
        "cy": np.array([r[3] for r in rows], np.float32),
        "xp": np.stack([r[4] for r in rows]),
        "yp": np.stack([r[5] for r in rows]),
        "ma": np.stack([r[6] for r in rows]),
    }
    return {k: torch.tensor(v) for k, v in vals.items()}


class Decoder(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.z = nn.Embedding(T, 32)
        nn.init.normal_(self.z.weight, 0, 0.15)
        self.freq = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        self.net = nn.Sequential(
            nn.Linear(58, 128),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 1),
        )
        self.logsig = nn.Parameter(torch.tensor(-1.0))

    def feat(self, x, y):
        q = [x, y]
        for k in self.freq:
            q += [
                torch.sin(np.pi * k * x),
                torch.cos(np.pi * k * x),
                torch.sin(np.pi * k * y),
                torch.cos(np.pi * k * y),
            ]
        return torch.stack(q, -1)

    def eta(self, t, x, y):
        return F.softplus(
            self.net(torch.cat([self.feat(x, y), self.z(t)], -1)).squeeze(-1)
        )

    def predict(self, t, cx, cy, xp, yp, ma, mode: str):
        if mode == "centroid":
            return self.eta(t, cx, cy)
        n, p = xp.shape
        ee = self.eta(
            t[:, None].expand(n, p).reshape(-1), xp.reshape(-1), yp.reshape(-1)
        ).reshape(n, p)
        return torch.log1p(
            (torch.expm1(torch.clamp(ee, max=7)) * ma).sum(1) / ma.sum(1)
        )


def fit_model(model, tr, mode: str, seed: int, steps: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1.4e-3, weight_decay=3e-6)
    rng = np.random.default_rng(seed)
    ntr = len(tr["t"])
    model.train()
    for step in range(steps):
        ii = torch.tensor(rng.integers(0, ntr, size=64))
        pred = model.predict(
            tr["t"][ii],
            tr["cx"][ii],
            tr["cy"][ii],
            tr["xp"][ii],
            tr["yp"][ii],
            tr["ma"][ii],
            mode,
        )
        sig = torch.exp(model.logsig)
        loss = (
            0.5 * torch.mean(((tr["y"][ii] - pred) / sig) ** 2)
            + model.logsig
            + 2e-6 * (model.z.weight**2).mean()
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        if (step + 1) % 300 == 0:
            print(f"step={step+1} loss={float(loss.detach()):.6f}", flush=True)


def predictive_draws(model, te, draws: int, chunk_size: int, seed_base: int):
    if draws % chunk_size != 0:
        raise ValueError("--draws must be a multiple of --chunk-size for the frozen protocol")
    model.train()  # keep dropout active
    n = len(te["t"])
    D = np.empty((draws, n), np.float32)
    cursor = 0
    nchunks = draws // chunk_size
    with torch.no_grad():
        for j in range(nchunks):
            torch.manual_seed(seed_base + j)
            np.random.seed(seed_base + j)
            k = chunk_size
            for st in range(0, n, 64):
                ed = min(st + 64, n)
                b = ed - st
                tt = te["t"][st:ed].repeat(k)
                xp = te["xp"][st:ed].repeat((k, 1))
                yp = te["yp"][st:ed].repeat((k, 1))
                ma = te["ma"][st:ed].repeat((k, 1))
                mu = model.predict(
                    tt,
                    te["cx"][st:ed].repeat(k),
                    te["cy"][st:ed].repeat(k),
                    xp,
                    yp,
                    ma,
                    "support",  # centroid-trained fit is evaluated as PostAgg
                ).reshape(k, b)
                eps = torch.randn_like(mu) * torch.exp(model.logsig)
                D[cursor : cursor + k, st:ed] = (mu + eps).numpy()
            cursor += k
            print(f"predictive draws: {cursor}/{draws}", flush=True)
    return D


def summarize(draws, target):
    mean = draws.mean(0)
    lo = np.quantile(draws, 0.05, axis=0)
    hi = np.quantile(draws, 0.95, axis=0)
    rmse = np.sqrt(np.mean((mean - target) ** 2))
    mae = np.mean(np.abs(mean - target))
    coverage = np.mean((target >= lo) & (target <= hi))
    width = np.mean(hi - lo)
    term1 = np.mean(np.abs(draws - target[None, :]), axis=0)
    sd = np.sort(draws, axis=0)
    n = draws.shape[0]
    coef = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    term2 = np.sum(coef * sd, axis=0) / (n * n)
    crps = np.mean(term1 - term2)
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "CRPS": float(crps),
        "Coverage90": float(coverage),
        "Width90": float(width),
    }, mean, lo, hi


def main():
    a = parse_args()
    torch.set_num_threads(2)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    outdir = Path(a.outdir)
    ckptdir = Path(a.checkpoint_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckptdir.mkdir(parents=True, exist_ok=True)

    A, water, xg, yg, T, _, _ = load_fields(a.input)
    tr = pack(make_blocks(A, water, xg, yg, shift=0))
    te = pack(make_blocks(A, water, xg, yg, shift=13 // 2))

    model = Decoder(T)
    ckpt = ckptdir / f"dropout_{a.mode}_bs13_seed{a.seed}.pt"
    if a.refit:
        fit_model(model, tr, a.mode, a.seed, a.steps)
        torch.save(model.state_dict(), ckpt)
    else:
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}. Use --refit to create it.")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    D = predictive_draws(model, te, a.draws, a.chunk_size, a.sampling_seed_base)
    target = te["y"].numpy()
    res, mean, lo, hi = summarize(D, target)
    res.update(
        {
            "method": "Support-aware" if a.mode == "support" else "Centroid-trained PostAgg",
            "draws": a.draws,
            "n_test": int(len(target)),
            "training_seed": a.seed,
            "sampling_seed_base": a.sampling_seed_base,
            "sigma": float(torch.exp(model.logsig).detach()),
        }
    )

    stem = "support" if a.mode == "support" else "centroid_postagg"
    pd.DataFrame({"target": target, "mean": mean, "lo90": lo, "hi90": hi}).to_csv(
        outdir / f"dropout_{stem}_bs13_S{a.draws}_predictions.csv", index=False
    )
    np.save(outdir / f"draws_{stem}_S{a.draws}.npy", D)
    (outdir / f"dropout_{stem}_bs13_S{a.draws}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
