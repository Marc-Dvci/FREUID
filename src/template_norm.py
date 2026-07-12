"""
Per-template score normalization.

The FREUID score pools every document into a single global operating point: APCER is read at the
threshold where exactly 1% of *all* bona-fides are rejected. That makes it brutally sensitive to
score-scale drift between document templates -- a detector can rank every template perfectly and
still score badly, because false alarms concentrated on one template push the global threshold up
and hide attacks on another.

This is not hypothetical. On the public test the baseline detector puts GUINEA/DL bona-fides near
0.02 while the other four templates' bona-fides sit near 0.25; the pooled 1%-BPCER threshold is then
set by the wrong population.

Fix: score each document by its rank *within its own template*, so every template contributes the
same score distribution to the pool. Both components of the FREUID score depend only on the ordering
of scores, so a within-cluster rank is a complete score -- no calibration step is needed.

Templates are discovered unsupervised (k-means over small thumbnails), never from labels or from a
fixed list of known types, so this works on the private test's unseen document types. It is
transductive over the test set only: no labels, no training data, no leakage.

Assumption: fraud prevalence is roughly comparable across templates. It holds in train (every type
is 40-50% fraud). `alpha` blends back toward the global ranking if that assumption is shaky:

    score = alpha * rank_within_cluster + (1 - alpha) * rank_global
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

THUMB = 16          # thumbnail edge used as the template signature
K_RANGE = range(2, 21)
SILHOUETTE_SAMPLE = 1500


def _signature(path, thumb=THUMB):
    """Small blurred RGB thumbnail: captures template layout + colour, ignores per-holder content."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros(thumb * thumb * 3, dtype=np.float32)
    img = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    v = img.reshape(-1)
    return v - v.mean()          # remove global brightness so exposure alone can't split a template


def signatures(paths, thumb=THUMB):
    return np.stack([_signature(p, thumb) for p in paths])


def choose_k(X, seed=0):
    """Pick the number of templates by silhouette.

    Getting this right matters more than it looks. Over-clustering is actively dangerous: split one
    template into enough pieces and the pieces start to align with the *fraud* signal rather than the
    template (a swapped portrait changes the thumbnail), and rank-normalizing inside a label-pure
    cluster would smear its scores across the whole range. Silhouette recovers exactly the 5 known
    document types on the FREUID train set, so we let it choose rather than fixing k by hand -- the
    private test has an unknown number of templates.
    """
    n = len(X)
    idx = np.random.default_rng(seed).choice(n, min(SILHOUETTE_SAMPLE, n), replace=False)
    Xs = X[idx]

    best_k, best_s = 2, -1.0
    for k in K_RANGE:
        if k >= len(Xs):
            break
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xs)
        s = silhouette_score(Xs, lab)
        if s > best_s:
            best_k, best_s = k, s
    return best_k, best_s


def cluster_templates(paths, k=None, seed=0):
    """Unsupervised template clusters. Returns (labels, k)."""
    X = signatures(paths)
    if k is None:
        k, _ = choose_k(X, seed=seed)
    k = min(k, len(paths))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(X), k


def rank01(x):
    """Rank-normalize to [0, 1]. Ties get distinct ranks; harmless for a rank-based metric."""
    order = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    return order / max(len(x) - 1, 1)


def normalize_per_template(scores, clusters, alpha=1.0, min_cluster=50):
    """Replace each score by its rank within its own template cluster.

    Clusters smaller than `min_cluster` are left on the global ranking -- a within-cluster rank over
    a handful of documents is noise, and a tiny cluster that happened to be label-pure would have its
    scores smeared across the whole [0, 1] range.
    """
    scores = np.asarray(scores, dtype=float)
    clusters = np.asarray(clusters)
    g = rank01(scores)
    out = g.copy()

    for c in np.unique(clusters):
        m = clusters == c
        if m.sum() < min_cluster:
            continue
        out[m] = rank01(scores[m])

    return alpha * out + (1.0 - alpha) * g
