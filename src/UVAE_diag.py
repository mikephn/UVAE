from src.UVAE_classes import *
from src.tools import *
from sklearn.mixture import GaussianMixture
import umap, re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plotsCallback(uv, doUmap=False, outFolder=None, n_samples=20000, dataMap=None, mapOf='2D', embeddings:list=None, um=None, ep=None):
    if outFolder is None:
        outFolder = os.path.dirname(uv.path) + '/'
    ensureFolder(outFolder)
    red2d = None
    if doUmap == False:
        reds = [r for r in uv.constraintsType(Projection) if r.latent_dim == 2]
        if len(reds):
            red2d = reds[0]
        else:
            print('No 2D projections found.')
            return
    if dataMap is None:
        inds = uv[mapOf].inds(validation=False)
        np.random.shuffle(inds)
        inds = inds[:n_samples]
        d = uv[mapOf].dataMap(inds)
    else:
        d = dataMap
    emb_panel_labels = np.concatenate([np.repeat(dt.name, len(d[dt])) for dt in d])
    if uv.history is not None and ep is None:
        ep = uv.history.epoch

    if type(embeddings) is list:
        embs = uv.mergedPredictMap(d, embeddings=embeddings)
        concat = []
        for data in embs:
            concat.append(embs[data])
            emb_cat = np.concatenate(concat)
    else:
        if embeddings is not None:
            emb = embeddings.predictMap(d, mean=True)
        else:
            emb = uv.predictMap(d, mean=True)
        emb_cat = np.vstack([emb[d] for d in emb])

    um_emb = None
    if doUmap:
        if um is None:
            um = umap.UMAP()
            um_emb = um.fit_transform(emb_cat)
        else:
            um_emb = um.transform(emb_cat)
    else:
        emb_2d = red2d.predictMap(d, stacked=True, mean=True)
        savePlot(emb_2d, emb_panel_labels, title='Panels (2D)', path=ensureFolder(outFolder + 'panels/') + 'panels-{}.png'.format(ep))
    if um_emb is not None:
        savePlot(um_emb, emb_panel_labels, title='Panels (UMAP)',
                 path=ensureFolder(outFolder) + 'panels-{}-umap.png'.format(ep))

    batch = uv.constraintsType(Normalization) + uv.constraintsType(MMD) + uv.constraintsType(Labeling)
    batch_ids = None
    if len(batch):
        batch_ids = ['-'] * len(emb_cat)
    for b in batch:
        b_cap = b.YsFromMap(d, undefined='-')
        b_cap = np.concatenate([b_cap[dt] for dt in b_cap])
        for i, bb in enumerate(b_cap):
            if bb != '-':
                batch_ids[i] = bb
    if batch_ids is not None:
        if doUmap == False:
            savePlot(emb_2d, batch_ids, title='Batches', path=ensureFolder(outFolder + 'batches/') + 'batches-{}.png'.format(ep))
        if um_emb is not None:
            savePlot(um_emb, batch_ids, title='Batches (UMAP)',
                     path=ensureFolder(outFolder) + 'batches-{}-umap.png'.format(ep))

    clsfs = [c for c in uv.constraints.values() if (type(c) is Classification)]
    for i, clsf in enumerate(clsfs):
        pred = clsf.predictMap(d, stacked=True)
        ref = None
        if doUmap == False:
            savePlot(emb_2d, pred, title=clsf.name, path=ensureFolder(outFolder + '{}/'.format(clsf.name)) + '{}-{}.png'.format(clsf.name, ep), refLabs=ref)
        if um_emb is not None:
            savePlot(um_emb, pred, title=clsf.name + ' (UMAP)',
                     path=ensureFolder(outFolder) + '{}-{}-umap.png'.format(clsf.name, ep),
                     refLabs=ref, legend=True)


def cachedUmap(path, dataFunc, n_dim=2):
    if os.path.exists(path):
        return unpickle(path)
    um = umap.UMAP(n_components=n_dim)
    dt = dataFunc()
    um.fit(dt)
    doPickle(um, path)
    return um


def gmClustering(uv, dm=None, emb=None, comps=10, name='GM', cov='full', embeddings:list=None, path=None):
    if path is not None:
        if fileExists(path):
            arch = unpickle(path)
            gmm = GaussianMixture(n_components=len(arch['means']), covariance_type=cov)
            gmm.precisions_cholesky_ = arch['cholesky']
            gmm.weights_ = arch['weights']
            gmm.means_ = arch['means']
            gmm.covariances_ = arch['covs']
            return gmm
    elif name in uv.constraints and uv[name].trained:
        return uv[name]
    if emb is None:
        if embeddings is None:
            emb = uv.predictMap(dm, mean=True)
        else:
            if type(embeddings) is list:
                emb = uv.mergedPredictMap(dm, embeddings)
            else:
                emb = embeddings.predictMap(dm, mean=True)
    emb_cat = stack(emb)
    gmm = GaussianMixture(n_components=comps, covariance_type=cov)
    print('Fitting GM:', path)
    clust_pred = gmm.fit_predict(emb_cat)
    if path is None:
        vals = repeatMasked(unstack(clust_pred, emb), dm, nullValue=-1)
        gm_clust = uv + Classification(Y=vals, name=name, nullLabel=-1, trainEmbedding=False)
        uv.train(maxEpochs=30, samplesPerEpoch=100000)
        return gm_clust
    else:
        arch = {'means': gmm.means_, 'covs': gmm.covariances_, 'weights': gmm.weights_, 'cholesky': gmm.precisions_cholesky_}
        doPickle(arch, path)
        return gmm


def leidenClustering(uv, dm=None, name='Leiden', fitEpochs=0)->{Data: np.ndarray}:
    emb = uv.predictMap(dm, mean=True, stacked=True)
    print('Leiden clustering...')
    import scanpy as sc
    t = time.time()
    ad = sc.AnnData(emb)
    sc.pp.neighbors(ad)
    sc.tl.leiden(ad)
    print('Clustering time: {}s'.format(int(time.time() - t)))
    clust_ld = np.array(ad.obs['leiden'].values, dtype=int)
    clust_dm = unstack(clust_ld, dm)
    if fitEpochs > 0:
        dm_res = expandPrediction(clust_dm, dm, nullValue=-1)
        clust = uv + Classification(Y=dm_res, name=name, nullLabel=-1, trainEmbedding=False)
        uv.train(maxEpochs=fitEpochs)
    return clust_dm


def filterClusters(uv, clustering, subset, subsample=0):
    all_data = uv.allDataMap()
    gm_cl_res = clustering.predictMap(all_data)
    subset_map = filterDataMap(all_data, gm_cl_res, subset)
    if subsample > 0:
        subset_map = subsampleDataMap(subset_map, subsample)
    return subset_map


def plotResult(uv, um, dm, result, path=None, title='', refLabs=None):
    emb = uv.predictMap(dm, mean=True)
    um_emb = um.transform(stack(emb))
    savePlot(um_emb, stack(result), path, title=title, refLabs=refLabs)


def plotPrediction(uv, um, dm, const, embeddings:list=None, path=None):
    if embeddings is None:
        emb = uv.predictMap(dm, mean=True)
    else:
        if type(embeddings) is list:
            emb = uv.mergedPredictMap(dm, embeddings)
        else:
            emb = embeddings.predictMap(dm, mean=True)
    um_emb = um.transform(stack(emb))
    p = const.predictMap(dm, mean=True, stacked=True)
    savePlot(um_emb, p, path, title=const.name)


def plotMarkers(uv, um, dm, markers=None, embeddings:list=None, path=None, interpolate=True):
    if embeddings is None:
        emb = uv.predictMap(dm, mean=True)
    else:
        if type(embeddings) is list:
            emb = uv.mergedPredictMap(dm, embeddings)
        else:
            emb = embeddings.predictMap(dm, mean=True)
    um_emb = um.transform(stack(emb))
    if markers is None:
        markers = np.unique(np.concatenate([d.channels for d in dm]))
    if interpolate:
        rec = stack(uv.reconstruct(dm, channels=markers))
    else:
        rec = uv.reconstruct(dm)
        stacked = []
        for d in rec:
            expanded = np.ones((len(rec[d]), len(markers)), dtype=float) * np.nan
            for ci, ch in enumerate(d.channels):
                if ch in markers:
                    col = list(markers).index(ch)
                    expanded[:, col] = rec[d][:, ci]
            stacked.append(expanded)
        rec = np.concatenate(stacked, axis=0)
    saveMarkerPlot(rec, um_emb, markers, path)


def plotLabels(uv, um, dm, labels, embeddings:list=None, path=None, title=''):
    if embeddings is None:
        emb = uv.predictMap(dm, mean=True)
    else:
        if type(embeddings) is list:
            emb = uv.mergedPredictMap(dm, embeddings)
        else:
            emb = embeddings.predictMap(dm, mean=True)
    um_emb = um.transform(stack(emb))
    p = stack(labels)
    savePlot(um_emb, p, path, title=title)


def savePlot(emb, labs, path, refLabs=None, title=None, size=0.1, quantile=0.99999, legend=True, legendLoc='best', firstBlack=False, axlim=0):
    if labs is None:
        labs = np.repeat(0, len(emb))
        legend = False
    if refLabs is None:
        refLabs = list(set(labs))
        refLabs.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', str(s))])

    xmin = np.quantile(emb[:, 0], 1-quantile)
    xmax = np.quantile(emb[:, 0], quantile)
    ymin = np.quantile(emb[:, 1], 1 - quantile)
    ymax = np.quantile(emb[:, 1], quantile)

    drop = np.zeros(len(emb), dtype=bool)
    for ie, e in enumerate(emb):
        if e[0] < xmin or e[0] > xmax or e[1] < ymin or e[1] > ymax:
            drop[ie] = True

    plt.figure(figsize=(8, 8))
    cm = plt.cm.hsv
    leg_cols = []
    leg_names = []
    samps = []
    cols = []
    for ci, l in enumerate(refLabs):
        inds = np.array([i for i in range(len(labs)) if labs[i] == l and not drop[i]])
        color = cm(float(ci)/len(refLabs))
        if firstBlack and ci == 0:
            color = 'k'
        if len(inds):
            e = emb[inds]
            samps.append(e)
            cols.extend([color] * len(e))
            leg_cols.append(color)
        else:
            leg_cols.append('#ffffff')
        leg_names.append(l)
    if len(samps) == 0:
        return
    samps = np.vstack(samps)
    perm = np.random.permutation(len(samps))
    plt.scatter(samps[perm, 0], samps[perm, 1], s=size, c=[cols[i] for i in perm])

    if legend and len(leg_names) < 31:
        handles = []
        for li in range(len(leg_names)):
            handles.append(mpatches.Patch(color=leg_cols[li], label=leg_names[li]))
        plt.legend(handles=handles, markerscale=2 / size, framealpha=0.5, loc=legendLoc)
    if axlim > 0:
        plt.xlim([-axlim, axlim])
        plt.ylim([-axlim, axlim])
    plt.title(title)
    if path is None:
        plt.show()
    else:
        fig = plt.gcf()
        fig.show()
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def saveMarkerPlot(X, emb, mks=None, path=None, quantile=0.999, skip=(), plotSize=4):
    xmin = np.quantile(emb[:, 0], 1-quantile)
    xmax = np.quantile(emb[:, 0], quantile)
    ymin = np.quantile(emb[:, 1], 1 - quantile)
    ymax = np.quantile(emb[:, 1], quantile)

    if type(X) is dict and mks is None:
        total_pts = np.sum([len(X[d]) for d in X])
        mks = list(np.unique(np.concatenate([d.channels for d in X])))
        arr = np.ones((total_pts, len(mks))) * np.nan
        loc = 0
        for d in X:
            d_len = len(X[d])
            for ci, ch in enumerate(d.channels):
                ch_col = mks.index(ch)
                arr[loc:loc+d_len, ch_col] = X[d][:, ci]
            loc += d_len
        X = arr

    if len(X.shape) == 1 and len(mks) == 1:
        X = np.expand_dims(X, -1)

    keep = np.ones(len(emb), dtype=bool)
    for ie, e in enumerate(emb):
        if e[0] < xmin or e[0] > xmax or e[1] < ymin or e[1] > ymax:
            keep[ie] = False

    perRow = min(4, len(mks))
    nRows = int(np.ceil((len(mks) - len(skip)) / perRow))
    fig = plt.figure(figsize=(int(plotSize*perRow)+1, int(plotSize*nRows)))
    r = 0
    c = 0
    cmap = plt.cm.cool
    for mi, mk in enumerate(mks):
        if mi in skip:
            continue
        m_vals = X[:, mi]
        valid = np.invert(np.isnan(m_vals))
        vmin = np.quantile(m_vals[valid], 0.01)
        vmax = np.quantile(m_vals[valid], 0.99)
        m_vals = np.clip(m_vals, vmin, vmax)
        m_vals -= np.nanmin(m_vals)
        m_vals /= np.nanmax(m_vals)
        colors = cmap(m_vals)
        ax = plt.subplot2grid((nRows, perRow), (r, c), rowspan=1, colspan=1)
        xv = emb[keep & valid, 0]
        yv = emb[keep & valid, 1]
        cols = colors[keep & valid]
        sc = ax.scatter(xv, yv, c=cols, s=0.2, cmap=cmap)
        ax.set_title(mk)
        if len(mks) == 1:
            cbaxes = fig.add_axes([0.2, 0.2, 0.3, 0.02])
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cb1 = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap, norm=norm,
                                            orientation='horizontal')
        c += 1
        if c == perRow:
            c = 0
            r += 1

    if path is None:
        plt.show()
    else:
        fig = plt.gcf()
        fig.show()
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def viewHyper(uv, path=None, names:list=None):
    if 'mango' in uv.archives and len(uv.archives['mango']) > 0:
        if names is None:
            names = list(uv.archives['mango'][0]['params'].keys())
        for name in names:
            losses = []
            vals = []
            for d in uv.archives['mango']:
                losses.append(d['loss'])
                vals.append(d['params'][name])
            plt.figure()
            plt.scatter(vals, losses)
            plt.title(uv.path)
            plt.ylabel('validation loss')
            plt.xlabel(name)
            if path is None:
                plt.show()
            else:
                ensureFolder(path)
                plt.savefig(path + '{}.png'.format(name), dpi=150, bbox_inches='tight')
            plt.close()
