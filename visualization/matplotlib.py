import matplotlib.pyplot as plt
import numpy as np


def plotlocs(
    x,
    y,
    z,
    alpha,
    maxptps,
    geom,
    feats=None,
    xlim=None,
    ylim=None,
    alim=None,
    zlim=None,
    which=slice(None),
    clip=True,
    suptitle=None,
    figsize=(8, 8),
    gs=1,
    cm=plt.cm.viridis,
):
    """Localization scatter plot
    Plots localizations (x, z, log y, log alpha) against the probe geometry,
    using max PTP to color the points.
    Arguments
    ---------
    x, y, z, alpha, maxptps : 1D np arrays of the same shape
    geom : n_channels x 2
    feats : optional, additional features to scatter
    *lim: optional axes lims if the default looks weird
    which : anything which can index x,y,z,alpha
        A subset of spikes to plot
    clip : bool
        If true (default), clip maxptps to the range 3-13 when coloring
        spikes
    """
    maxptps = maxptps[which]
    nmaxptps = 0.1
    cmaxptps = maxptps
    if clip:
        nmaxptps = 0.25 + 0.74 * (maxptps - maxptps.min()) / (
            maxptps.max() - maxptps.min()
        )
        cmaxptps = np.clip(maxptps, 3, 13)

    x = x[which]
    y = y[which]
    alpha = alpha[which]
    z = z[which]

    nfeats = 0
    if feats is not None:
        nfeats = feats.shape[1]

    fig, axes = plt.subplots(1, 3 + nfeats, sharey=True, figsize=figsize)
    aa, ab, ac = axes[:3]
    aa.scatter(x, z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
    aa.scatter(geom[:, 0], geom[:, 1], color="orange", marker="s", s=gs)
    logy = np.log(y)
    ab.scatter(logy, z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
    loga = np.log(alpha)
    ac.scatter(loga, z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
    aa.set_ylabel("z")
    aa.set_xlabel("x")
    ab.set_xlabel("$\\log y$")
    ac.set_xlabel("$\\log \\alpha$")
    if xlim is None:
        aa.set_xlim(np.percentile(x, [0, 100]) + [-10, 10])
    else:
        aa.set_xlim(xlim)
    if ylim is None:
        ab.set_xlim(np.percentile(logy, [0, 100]))
    else:
        ab.set_xlim(ylim)
    # ab.set_xlim([-0.5, 6])
    if alim is None:
        ac.set_xlim(np.percentile(loga, [0, 100]))
    else:
        ac.set_xlim(alim)

    if suptitle:
        fig.suptitle(suptitle, y=0.95)

    if feats is not None:
        for ax, f in zip(axes[3:], feats.T):
            ax.scatter(f[which], z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
            ax.set_xlim(np.percentile(f, [0, 100]))

    if zlim is None:
        aa.set_ylim([z.min() - 10, z.max() + 10])
    else:
        aa.set_ylim(zlim)
