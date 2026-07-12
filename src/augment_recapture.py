"""
Print-and-capture ("analog hole") simulation.

Train is ~99.97% fully digital but the test set emphasizes print-and-capture and screen
recapture. A model trained naively will key on digital-only artifacts that vanish after
recapture. This module synthesizes the physical pipeline so the model learns features that
survive it, and so digital-vs-recaptured cannot be used as a fraud shortcut (it is applied to
both classes).

Design notes:
  * Pure numpy/cv2 on uint8 RGB images -> stable across albumentations versions. The dataset
    applies only Resize/Normalize/ToTensor from albumentations afterwards.
  * Ops are modular and individually probabilistic; `RecaptureSimulator` composes them.
  * The "macro" path (down-up resample + double JPEG + slight blur + mild color cast) is the
    core of recapture and is what most reliably closes the gap; texture/moire/glare add realism.
"""
import numpy as np
import cv2


# ---------- individual ops (each takes/returns uint8 RGB HxWx3) ----------

def jpeg_recompress(img, quality):
    enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def down_up_resample(img, scale, interp_down=cv2.INTER_AREA, interp_up=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=interp_down)
    return cv2.resize(small, (w, h), interpolation=interp_up)


def add_moire(img, rng, freq_range=(0.05, 0.35), angle_range=(0, 180), strength=(0.04, 0.12)):
    """Additive sinusoidal interference, as left by photographing a screen."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = img.astype(np.float32)
    for _ in range(rng.integers(1, 3)):
        f = rng.uniform(*freq_range)
        ang = np.deg2rad(rng.uniform(*angle_range))
        s = rng.uniform(*strength) * 255.0
        grating = np.sin(2 * np.pi * f * (xx * np.cos(ang) + yy * np.sin(ang)))
        out += s * grating[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_halftone(img, rng, cell=(3, 6), mix=(0.15, 0.4)):
    """Crude print-halftone screen: blend a dot-screened version over the image."""
    c = int(rng.integers(cell[0], cell[1] + 1))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    screen = (0.5 + 0.5 * np.sin(np.pi * xx / c) * np.sin(np.pi * yy / c))
    dots = (gray > screen).astype(np.float32)[..., None]
    dotted = (dots * 255).astype(np.uint8).repeat(3, axis=2)
    m = rng.uniform(*mix)
    return np.clip((1 - m) * img + m * dotted, 0, 255).astype(np.uint8)


def add_glare(img, rng, n=(1, 2), intensity=(0.25, 0.6)):
    """Bright elliptical specular highlight(s), as from a flash/overhead light on a print."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    for _ in range(int(rng.integers(n[0], n[1] + 1))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ax, ay = rng.uniform(0.1, 0.4) * w, rng.uniform(0.1, 0.4) * h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
        mask = np.exp(-d) * (rng.uniform(*intensity) * 255.0)
        out += mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_vignette(img, rng, strength=(0.2, 0.5)):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    r = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    s = rng.uniform(*strength)
    mask = 1.0 - s * np.clip(r, 0, 1) ** 2
    return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def paper_texture(img, rng, strength=(0.03, 0.10)):
    """Multiplicative low-freq paper grain + faint fibers."""
    h, w = img.shape[:2]
    noise = rng.normal(0, 1, (h // 4 + 1, w // 4 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    s = rng.uniform(*strength)
    return np.clip(img.astype(np.float32) * (1 + s * noise[..., None]), 0, 255).astype(np.uint8)


def color_cast(img, rng, shift=(-18, 18), gain=(0.92, 1.08)):
    """White-balance / illuminant shift from a different capture device."""
    out = img.astype(np.float32)
    out = out * rng.uniform(gain[0], gain[1], 3)[None, None, :]
    out = out + rng.uniform(shift[0], shift[1], 3)[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def sensor_noise(img, rng, read=(2.0, 8.0), shot=(0.0, 0.02)):
    """Poisson(shot) + Gaussian(read) noise of a camera sensor."""
    out = img.astype(np.float32)
    out = out + rng.normal(0, rng.uniform(*read), img.shape)
    sh = rng.uniform(*shot)
    if sh > 0:
        out = out + rng.normal(0, 1, img.shape) * np.sqrt(np.clip(out, 0, None)) * sh
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(img, rng, max_k=3):
    k = int(rng.integers(1, max_k + 1)) * 2 + 1
    if rng.random() < 0.5:
        return cv2.GaussianBlur(img, (k, k), 0)
    # motion blur
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    ang = rng.uniform(0, 180)
    M = cv2.getRotationMatrix2D((k / 2, k / 2), ang, 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)


def perspective_warp(img, rng, jitter=0.04):
    h, w = img.shape[:2]
    j = jitter
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-j, j, (4, 2)) * np.float32([w, h])
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


# ---------- composer ----------

class RecaptureSimulator:
    """Probabilistic composition of the ops above.

    strength in {"light","medium","heavy"} scales op probabilities and magnitudes.
    `force_macro` always applies the core recapture macro (down-up + double JPEG + blur)
    -- use it to build the deterministic recapture stress-evaluation set.
    """
    PRESETS = {
        "light":  dict(p_macro=0.5, p_aux=0.25, scale=(0.6, 0.9), q=(70, 95)),
        "medium": dict(p_macro=0.7, p_aux=0.4,  scale=(0.45, 0.8), q=(50, 90)),
        "heavy":  dict(p_macro=0.9, p_aux=0.55, scale=(0.3, 0.7),  q=(35, 80)),
    }

    def __init__(self, strength="medium", force_macro=False, seed=None):
        self.cfg = self.PRESETS[strength]
        self.force_macro = force_macro
        self.seed = seed

    def __call__(self, image, rng=None):
        rng = rng or np.random.default_rng(self.seed)
        cfg = self.cfg
        img = image

        if rng.random() < 0.5:
            img = perspective_warp(img, rng)

        # core recapture macro: resample through a lower resolution + double JPEG (+ blur)
        if self.force_macro or rng.random() < cfg["p_macro"]:
            img = down_up_resample(img, rng.uniform(*cfg["scale"]))
            img = jpeg_recompress(img, rng.integers(*cfg["q"]))
            if rng.random() < 0.6:
                img = blur(img, rng, max_k=2)
            img = jpeg_recompress(img, rng.integers(*cfg["q"]))

        pa = cfg["p_aux"]
        if rng.random() < pa:
            img = add_moire(img, rng) if rng.random() < 0.5 else add_halftone(img, rng)
        if rng.random() < pa:
            img = add_glare(img, rng)
        if rng.random() < pa:
            img = add_vignette(img, rng)
        if rng.random() < pa:
            img = paper_texture(img, rng)
        if rng.random() < 0.7:
            img = color_cast(img, rng)
        if rng.random() < 0.7:
            img = sensor_noise(img, rng)
        if rng.random() < 0.4:
            img = jpeg_recompress(img, rng.integers(*cfg["q"]))  # final capture compression
        return img


if __name__ == "__main__":
    import sys, glob, os
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "the-freuid-challenge-2026-ijcai-ecai"
    files = glob.glob(os.path.join(data_dir, "train", "train", "*.jpeg"))[:3]
    rng = np.random.default_rng(0)
    os.makedirs("aug_preview", exist_ok=True)
    for f in files:
        bgr = cv2.imread(f)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for strength in ("light", "medium", "heavy"):
            sim = RecaptureSimulator(strength)
            out = sim(rgb, rng)
            name = f"aug_preview/{os.path.splitext(os.path.basename(f))[0]}_{strength}.jpg"
            cv2.imwrite(name, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            print("wrote", name, out.shape, out.dtype)
    print("done")
