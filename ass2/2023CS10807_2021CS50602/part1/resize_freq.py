import os
import sys
from typing import Tuple

import cv2
import numpy as np


def load_gray(p: str) -> np.ndarray:
    i = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if i is None:
        raise FileNotFoundError(f"Could not load image: {p}")
    return i.astype(np.float32) / 255.0


def save_gray(p: str, i: np.ndarray) -> None:
    o = np.clip(i * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(p, o)


def naive_resample(x: np.ndarray, s: float) -> np.ndarray:
    h, w = x.shape
    nh = max(1, int(np.round(h * s)))
    nw = max(1, int(np.round(w * s)))

    if s >= 1.0:
        ys = (np.arange(nh) + 0.0) / s
        xs = (np.arange(nw) + 0.0) / s
        yv, xv = np.meshgrid(ys, xs, indexing="xy")
        y0 = np.floor(yv).astype(int)
        x0 = np.floor(xv).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        wy = yv - y0
        wx = xv - x0
        top = (1 - wx) * x[y0, x0] + wx * x[y0, x1]
        bot = (1 - wx) * x[y1, x0] + wx * x[y1, x1]
        res = (1 - wy) * top + wy * bot
        return res.T if False else res

    else:
        ys = (np.arange(nh) / s).astype(int)
        xs = (np.arange(nw) / s).astype(int)
        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)
        return x[np.ix_(ys, xs)]


def ideal_lowpass_prefilter(x: np.ndarray, s: float) -> np.ndarray:
    if s >= 1.0:
        return x.copy()

    def p2(n: int) -> int:
        return 1 << (n - 1).bit_length()

    def f1(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        n = x.shape[0]
        if n == 1:
            return x
        e = f1(x[0::2])
        o = f1(x[1::2])
        f = np.exp(-2j * np.pi * np.arange(n) / n)
        return np.concatenate([e + f[: n // 2] * o,
                               e + f[n // 2:] * o])

    def i1(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.complex128)
        n = X.shape[0]
        return np.conjugate(f1(np.conjugate(X))) / n

    def f2(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.complex128)
        h, w = a.shape
        H = p2(h)
        W = p2(w)
        Ap = np.zeros((H, W), dtype=np.complex128)
        Ap[:h, :w] = a
        for r in range(H):
            Ap[r, :] = f1(Ap[r, :])
        for c in range(W):
            Ap[:, c] = f1(Ap[:, c])
        return Ap, H, W

    def i2(Ap: np.ndarray, H: int, W: int) -> np.ndarray:
        for c in range(W):
            Ap[:, c] = i1(Ap[:, c])
        for r in range(H):
            Ap[r, :] = i1(Ap[r, :])
        return np.real(Ap)

    def fs(A: np.ndarray) -> np.ndarray:
        return np.roll(np.roll(A, A.shape[0] // 2, axis=0), A.shape[1] // 2, axis=1)

    def ifs(A: np.ndarray) -> np.ndarray:
        return np.roll(np.roll(A, - (A.shape[0] // 2), axis=0), - (A.shape[1] // 2), axis=1)

    h, w = x.shape
    Fp, Hp, Wp = f2(x)
    Fs = fs(Fp)

    fy = (np.arange(-Hp // 2, -Hp // 2 + Hp) / Hp)
    fx = (np.arange(-Wp // 2, -Wp // 2 + Wp) / Wp)
    FX, FY = np.meshgrid(fx, fy)

    c = 0.5 * s
    m = (np.sqrt(FX * FX + FY * FY) <= c).astype(np.float64)

    Ff = Fs * m
    Fi = ifs(Ff)
    r = i2(Fi, Hp, Wp)
    return r[:h, :w]


def resize_and_save(i: np.ndarray, s: float, d: str, b: str) -> None:
    os.makedirs(d, exist_ok=True)
    h, w = i.shape
    n = naive_resample(i, s)
    save_gray(os.path.join(d, f"{b}_naive_k{int(1/s) if s<1 else int(s)}.png"), n)
    fp = ideal_lowpass_prefilter(i, s)
    fr = naive_resample(fp, s)
    save_gray(os.path.join(d, f"{b}_filtered_k{int(1/s) if s<1 else int(s)}.png"), fr)
    def z(a: np.ndarray) -> np.ndarray:
        nh, nw = a.shape
        fy = nh / float(h) if h > 0 else 1.0
        fx = nw / float(w) if w > 0 else 1.0
        ys = np.floor(np.arange(h) * fy).astype(int)
        xs = np.floor(np.arange(w) * fx).astype(int)
        ys = np.clip(ys, 0, nh - 1)
        xs = np.clip(xs, 0, nw - 1)
        return a[np.ix_(ys, xs)]
    bn = z(n)
    bf = z(fr)
    save_gray(os.path.join(d, f"{b}_naive_back_k{int(1/s) if s<1 else int(s)}.png"), bn)
    save_gray(os.path.join(d, f"{b}_filtered_back_k{int(1/s) if s<1 else int(s)}.png"), bf)


def main():
    p = os.path.dirname(__file__)
    ip = os.path.join(p, "barbara.bmp")
    od = os.path.join(p, "results")
    i = load_gray(ip)
    ss = [0.5, 0.25, 0.125]
    for s in ss:
        print(f"Processing s={s}")
        resize_and_save(i, s, od, "barbara")


if __name__ == "__main__":
    main()
