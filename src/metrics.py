"""
FREUID Score and its components.

The competition metric (lower is better):

    g_audet = 1 - AuDET
    g_apcer = 1 - APCER@1%BPCER
    FREUID  = 1 - 2 * g_audet * g_apcer / (g_audet + g_apcer)   # 1 - harmonic mean

where, with label 1 = attack (positive) and label 0 = bona-fide:
    BPCER = bona-fide classified as attack   (= false positive rate)
    APCER = attack classified as bona-fide   (= false negative rate)
    AuDET = area under the Detection Error Trade-off curve (APCER vs BPCER).

Both AuDET and APCER@1%BPCER are in [0, 1], lower is better, so the final
FREUID score is in [0, 1] and lower is better.
"""
import numpy as np
from sklearn.metrics import det_curve


_trapz = getattr(np, "trapezoid", np.trapz)  # numpy>=2 renamed trapz->trapezoid


def _det(y_true, y_score):
    """sklearn DET curve, re-sorted so BPCER (fpr) is ascending.

    NOTE: sklearn's det_curve returns BPCER in *descending* order (1.0 -> 0.0) with APCER (fnr)
    ascending. We sort by BPCER ascending so APCET-vs-BPCER integration and interpolation are
    well-defined. After sorting, APCER is non-increasing as BPCER increases.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    bpcer, apcer, _ = det_curve(y_true, y_score, pos_label=1)
    order = np.argsort(bpcer, kind="stable")
    return bpcer[order], apcer[order]


def calculate_audet(y_true, y_score):
    """Area under the DET curve (APCER as a function of BPCER), integrated over BPCER in [0, 1].

    Lower is better. A perfect detector -> ~0, random -> ~0.5.
    """
    bpcer, apcer = _det(y_true, y_score)

    # Pin the curve to the full [0, 1] BPCER range so the area is comparable and truly bounded.
    if bpcer[0] > 0.0:
        bpcer = np.concatenate([[0.0], bpcer])
        apcer = np.concatenate([[apcer[0]], apcer])   # at BPCER=0 keep the worst (highest) APCER
    if bpcer[-1] < 1.0:
        bpcer = np.concatenate([bpcer, [1.0]])
        apcer = np.concatenate([apcer, [0.0]])         # at BPCER=1 everything flagged -> APCER=0

    return float(_trapz(apcer, bpcer))


def calculate_apcer_at_bpcer(y_true, y_score, target_bpcer=0.01):
    """APCER at a fixed BPCER (default 1%), linearly interpolated at exactly target_bpcer.

    Because BPCER is non-decreasing and APCER non-increasing along the DET curve, we can
    interpolate APCER as a function of BPCER. If target_bpcer is below the lowest achievable
    BPCER (too few bona-fide samples to resolve 1%), we fall back to the APCER at the lowest
    BPCER point -- the strictest reachable operating point.
    """
    bpcer, apcer = _det(y_true, y_score)  # bpcer ascending, apcer non-increasing

    if target_bpcer <= bpcer[0]:
        return float(apcer[0])
    if target_bpcer >= bpcer[-1]:
        return float(apcer[-1])
    return float(np.interp(target_bpcer, bpcer, apcer))


def calculate_freuid_score(y_true, y_score, target_bpcer=0.01):
    """Return (freuid_score, audet, apcer@target). Lower freuid_score is better."""
    audet = calculate_audet(y_true, y_score)
    apcer = calculate_apcer_at_bpcer(y_true, y_score, target_bpcer=target_bpcer)

    g_audet = 1.0 - audet
    g_apcer = 1.0 - apcer

    denom = g_audet + g_apcer
    hm = 0.0 if denom == 0 else 2.0 * g_audet * g_apcer / denom
    freuid_score = 1.0 - hm
    return freuid_score, audet, apcer


def freuid_report(y_true, y_score, target_bpcer=0.01):
    """Convenience dict for logging."""
    f, a, p = calculate_freuid_score(y_true, y_score, target_bpcer)
    return {"freuid": f, "audet": a, "apcer@%.0f%%bpcer" % (target_bpcer * 100): p}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Perfect separation -> FREUID ~ 0
    y = np.array([0] * 500 + [1] * 500)
    s_perfect = np.concatenate([rng.uniform(0.0, 0.4, 500), rng.uniform(0.6, 1.0, 500)])
    print("perfect :", freuid_report(y, s_perfect))

    # Random scores -> FREUID ~ 1 (AuDET ~0.5, APCER@1% ~ near 1)
    s_random = rng.uniform(0, 1, 1000)
    print("random  :", freuid_report(y, s_random))

    # Decent-but-imperfect overlap
    s_ok = np.concatenate([rng.normal(0.35, 0.15, 500), rng.normal(0.65, 0.15, 500)]).clip(0, 1)
    print("decent  :", freuid_report(y, s_ok))

    # Sanity: interpolation hits exactly target on a controllable case
    # 100 bona-fide uniform in [0,1], 100 attacks all at 1.0 -> at BPCER=1% threshold ~0.99,
    # APCER should be ~0 (all attacks above threshold).
    yy = np.array([0] * 100 + [1] * 100)
    ss = np.concatenate([np.linspace(0, 0.98, 100), np.full(100, 1.0)])
    print("clean-hi:", freuid_report(yy, ss))
