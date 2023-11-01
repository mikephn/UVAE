from src.UVAE_classes import *
from src.tools import *
from sklearn.mixture import GaussianMixture
import umap, re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plotsCallback(uv, doUmap=False, outFolder=None, n_samples=20000, dataMap=None, showSamplesOf='2D', plotLabels:list=None, um=None, ep=None):
    """
    Example callback function for visualising embeddings each epoch using either UMAP or 2D projections.

    Visualization of latent space is created either using UMAP, or a model Projection constraint.
    Categorical labels are then shown on the embedding.

    Parameters:
    - uv: The UVAE model to be used for generating the embeddings.
    - doUmap (bool, optional): Whether to use UMAP for visualization. If False, uses available 2D projections. Default is False.
    - outFolder (str, optional): Directory where to save the plots. If None, uses the directory of `uv.path`.
    - n_samples (int, optional): Number of samples to visualize. Default is 20000.
    - dataMap (dict, optional): Predefined data mapping. If None, it's generated based on the `mapOf` parameter.
    - showSamplesOf (str, optional): Specifies the data source for mapping. Default is '2D'.
    - plotLabels (list, optional): List of labelings to visualize. If None, shows all included categorical labelings.
    - um (UMAP object, optional): Pretrained UMAP model. If None and `doUmap` is True, a new UMAP model will be trained.
    - ep (int, optional): Current epoch number. If None and `uv.history` is not None, uses the last epoch from the history.

    Returns:
    None

    Dependencies:
    This function requires `matplotlib`, `UMAP` if used.
    """
    # Create output folder if not given
    if outFolder is None:
        outFolder = os.path.dirname(uv.path) + '/'
    ensureFolder(outFolder)
    red2d = None

    if doUmap == False:
        # Find a 2D projection if no UMAP projection is chosen
        reds = [r for r in uv.constraintsType(Projection) if r.latent_dim == 2]
        if len(reds):
            red2d = reds[0]
        else:
            print('No 2D projections found.')
            return

    # Select samples to plot
    if dataMap is None:
        if uv[showSamplesOf] is None:
            print('No {} constraint found in UVAE. Specify dataMap or a constraint to use as a sample source.'.format(showSamplesOf))
            return
        inds = uv[showSamplesOf].inds(validation=False)
        np.random.shuffle(inds)
        inds = inds[:n_samples]
        d = uv[showSamplesOf].dataMap(inds)
    else:
        d = dataMap

    # Create labels for data streams
    emb_panel_labels = np.concatenate([np.repeat(dt.name, len(d[dt])) for dt in d])

    # Get training epoch
    if uv.history is not None and ep is None:
        ep = uv.history.epoch

    # Get latent embedding
    emb = uv.predictMap(d, mean=True)

    if doUmap:
        # Use UMAP
        if um is None: # train UMAP
            um = umap.UMAP()
            emb_2d = um.fit_transform(emb)
        else: # use provided UMAP
            emb_2d = um.transform(emb)
    else:
        # Use 2D projection from the model
        emb_2d = red2d.predictMap(d, stacked=True, mean=True)

    # Save panel plot
    savePlot(emb_2d, emb_panel_labels, title='Panels',
             path=ensureFolder(outFolder + 'panels/') + 'panels-{}.png'.format(ep))

    # Save remaining label plots
    clsfs = [c for c in uv.constraints.values() if isinstance(c, Labeling)]
    for i, clsf in enumerate(clsfs):
        if plotLabels is not None:
            if clsf.name not in plotLabels:
                continue
        pred = clsf.predictMap(d, stacked=True)
        savePlot(emb_2d, pred, title=clsf.name, path=ensureFolder(outFolder + '{}/'.format(clsf.name)) + '{}-{}.png'.format(clsf.name, ep))


def savePlot(emb, labs, path, refLabs=None, title=None, size=0.1, quantile=0.99999, legend=True, legendLoc='best', firstBlack=False, axlim=0, lims=None, dpi=150):
    """
    Create and save a scatter plot of embedded data with optional legend and customized appearance.

    This function provides a visualization of embedded data points, with color coding based on provided labels.
    Outliers, defined based on a specified quantile, are dropped from the visualization.

    Parameters:
    - emb (np.ndarray): The embedded data of shape (n_samples, n_dimensions).
    - labs (list or np.ndarray): Labels for each embedded data point.
    - path (str): File path to save the plot. If None, the plot is displayed.
    - refLabs (list, optional): Reference labels used to define the color order in the legend.
                                If not provided, it's inferred from the unique values in `labs`.
    - title (str, optional): Title for the plot.
    - size (float, optional): Size of each data point in the scatter plot. Default is 0.1.
    - quantile (float, optional): Quantile threshold to identify and drop outliers. Default is 0.99999.
    - legend (bool, optional): Whether to display a legend. Default is True.
    - legendLoc (str, optional): Location of the legend on the plot. Default is 'best'.
    - firstBlack (bool, optional): If True, the first label in the reference labels will be colored black. Default is False.
    - axlim (float, optional): Axis limit for both x and y axes. Default is 0, which means no limit.
    - lims (list or tuple, optional): Specific axis limits as [x_min, x_max, y_min, y_max]. Overrides `axlim` if provided.
    - dpi (int, optional): Dots per inch for the saved figure. Default is 150.

    Returns:
    None

    Notes:
    The function uses `matplotlib` for visualization. Depending on the provided labels and the quantile value,
    outliers are identified and excluded from the plot.

    Dependencies:
    This function requires the `matplotlib` library for plotting.
    """

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
    if lims is not None:
        plt.xlim([lims[0], lims[1]])
        plt.ylim([lims[2], lims[3]])
    plt.title(title)
    if path is None:
        plt.show()
    else:
        fig = plt.gcf()
        fig.show()
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()


def cachedUmap(path, dataFunc, n_dim=2, **kwargs):
    """
    Apply UMAP dimensionality reduction, utilizing cached results if available.

    This function attempts to load a precomputed UMAP model from the specified path. If no model exists at the path,
    UMAP is applied to the data retrieved using the provided data function, and the resulting model is cached for
    future use.

    Parameters:
    - path (str): The file path to either load a cached UMAP model or save the computed UMAP model.
    - dataFunc (callable): A function that returns the data to be used for UMAP when called with the provided keyword arguments.
    - n_dim (int, optional): The number of dimensions for the UMAP embedding. Default is 2.
    - **kwargs: Arbitrary keyword arguments that are passed to the `dataFunc`.

    Returns:
    - UMAP object: A trained UMAP model, either loaded from cache or trained on the provided data.

    Note:
    It is essential that the `dataFunc` returns data in a consistent format and structure, especially when using
    cached results. Any changes to the data or its structure may require clearing the cached UMAP model and
    re-computing.

    Dependencies:
    This function requires the `umap.UMAP` class for dimensionality reduction.
    """
    if os.path.exists(path):
        return unpickle(path)
    um = umap.UMAP(n_components=n_dim)
    dt = dataFunc(**kwargs)
    um.fit(np.array(dt, dtype=float))
    doPickle(um, path)
    return um


def gmmClustering(X, path, B=None, comps=[10], cov='full', subsample=100000):
    """
    Apply Gaussian Mixture Model (GMM) clustering to the provided data.

    This function clusters the data using the Gaussian Mixture Model. It can either cluster the entire data set as a whole
    or cluster each batch (if provided) separately. The GMM models are persisted to a specified path for future use.

    Parameters:
    - X (array-like): A numerical 2D array-like structure representing the data. Rows correspond to data points, and columns
      correspond to features/dimensions.
    - path (str): Path to store or load the trained GMM models.
    - B (array-like, optional): Batch labels corresponding to each data point in `X`. If provided, the function will cluster
      each batch separately. Default is None, which clusters all data together.
    - comps (list or int): List of integers specifying the number of components (clusters) for the GMM. If an integer is
      provided, it is converted to a list with a single value. Default is [10].
    - cov (str): Type of covariance matrix to use for the GMM. It can be one of {'full', 'tied', 'diag', 'spherical'}.
      Default is 'full'.
    - subsample (int): Number of samples to randomly select from `X` for training the GMM. If set to 0 or a number greater
      than the number of samples in `X`, the entire dataset `X` is used. Default is 100000.

    Returns:
    - list: A list of 1D numpy arrays representing cluster assignments for each component count in `comps`.

    Note:
    The function prints messages about the progress of GMM training, especially when dealing with multiple batches or
    components. If GMM models already exist at the specified path, they are loaded and used for prediction without
    re-training.

    Dependencies:
    This function requires the `sklearn.mixture.GaussianMixture` class for clustering.
    """
    if fileExists(path):
        clst = unpickle(path)
        print('Loaded GMM ({} models).'.format(len(clst)))
    else:
        clst = {}
    if type(comps) is int:
        comps = [comps]
    results = []
    for ci, n_c in enumerate(comps):
        if B is None:
            # cluster the data together
            if ci not in clst:
                gmm = GaussianMixture(n_components=n_c, covariance_type=cov)
                if subsample > 0:
                    sample = X[np.random.permutation(len(X))[:subsample]]
                else:
                    sample = X
                print('Fitting GMM ({} comps).'.format(n_c))
                gmm.fit(sample)
                clst[ci] = gmm
            results.append(clst[ci].predict(X))
        else:
            # cluster each batch independently
            if ci not in clst:
                clst[ci] = {}
            batches = sorted(list(set(B)))
            res = np.zeros(len(X), dtype=int)
            for bi, b in enumerate(batches):
                mask = B == b
                b_X = X[mask]
                if b not in clst[ci]:
                    gmm = GaussianMixture(n_components=n_c, covariance_type=cov)
                    if subsample > 0:
                        sample = b_X[np.random.permutation(len(b_X))[:subsample]]
                    else:
                        sample = b_X
                    print('Fitting GMM to batch {} ({} comps).'.format(b, n_c))
                    gmm.fit(sample)
                    clst[ci][b] = gmm
                b_res = clst[ci][b].fit_predict(b_X)
                res[mask] = b_res + int(bi * n_c)
            results.append(res)
    doPickle(clst, path)
    return results


def leidenClustering(emb)->np.ndarray:
    """
    Apply Leiden clustering algorithm to a given embedding.

    This function prints the duration taken for clustering and the number of identified clusters.

    Parameters:
    - emb (array-like): A numerical 2D array-like structure representing the data embedding. Rows correspond to data
      points and columns correspond to embedding dimensions.

    Returns:
    - np.ndarray: A 1D array of cluster labels assigned to each data point in the embedding.

    Dependencies:
    This function requires the `scanpy` library.
    """
    print('Leiden clustering...')
    import scanpy as sc
    t = time.time()
    ad = sc.AnnData(emb)
    sc.pp.neighbors(ad)
    sc.tl.leiden(ad)
    clust_ld = np.array(ad.obs['leiden'].values, dtype=int)
    print('Clustering time: {}s, {} clusters.'.format(int(time.time() - t), len(set(clust_ld))))
    return clust_ld


def classNormalizationMask(batches, labels):
    """
    Generate a mask for class normalization.

    The function determines the minimum count of each class across different batches.
    It then generates a binary mask where each class in each batch is subsampled
    to match the minimum count across batches.

    Parameters:
    - batches (array-like): An array indicating the batch each sample belongs to.
    - labels (array-like): An array indicating the class label of each sample.

    Returns:
    - array-like (bool): A binary mask indicating which samples are to be included after normalization.
    """
    cts = {}
    bs = list(set(batches))
    ls = list(set(labels))
    for c in ls:
        b_cts = []
        for b in bs:
            mask = np.logical_and(labels == c, batches == b)
            b_cts.append(np.sum(mask))
        cts[c] = b_cts
    min_counts = {c: np.min(cts[c]) for c in cts}
    valid_mask = np.zeros(len(labels), dtype=bool)
    for c in cts:
        if min_counts[c] > 0:
            for b in bs:
                mask = np.logical_and(labels == c, batches == b)
                inds = np.arange(len(mask))[mask]
                inds_subsample = np.random.permutation(inds)[:min_counts[c]]
                valid_mask[inds_subsample] = True
    return valid_mask


def calculateLISI(emb, batches, name, outFolder, classes, normClass=False, perplexity=30, scoreFilename='LISI_scores.csv'):
    """
    Calculate LISI scores for given embeddings and metadata.

    The function computes the LISI score for the given embeddings with respect to batches and classes.
    If harmonypy library is available, it uses it to compute the LISI scores. Otherwise, it falls back
    to using an R script.

    Parameters:
    - emb (array-like): The embeddings for which LISI is to be calculated.
    - batches (array-like): An array indicating the batch each sample belongs to.
    - name (str): Name of the current analysis (used for saving results).
    - outFolder (str): Directory path where the results should be saved.
    - classes (array-like or dict): Class labels for each sample.
    - normClass (bool, optional): Whether to normalize classes. Defaults to False.
    - perplexity (int, optional): Perplexity value to use for LISI computation. Defaults to 30.
    - scoreFilename (str, optional): Name of the file where LISI scores should be saved. Defaults to 'LISI_scores.csv'.

    Returns:
    - dict: A dictionary containing LISI scores for batches and classes.
    """
    types = ['batch']
    meta = batches
    if type(classes) is dict:
        for k in classes:
            types.append(k)
            meta = np.column_stack([meta, classes[k]])
    else:
        types.append('class')
        meta = np.column_stack([meta, classes])
    if normClass and type(classes) is np.ndarray:
        normMask = classNormalizationMask(batches, classes)
        meta = meta[normMask]
        emb = emb[normMask]
    try:
        import harmonypy.lisi
        meta_pd = pd.DataFrame(meta, columns=types)
        result = harmonypy.lisi.compute_lisi(emb, meta_pd, types, perplexity)
        res_med = np.median(result, axis=0)
        score_row = pd.DataFrame([name]+list(res_med)).T
        score_row.columns = ['name']+types
        if fileExists(outFolder + scoreFilename):
            scores = pd.read_csv(outFolder + scoreFilename)
            scores = pd.concat([scores, score_row], axis=0, ignore_index=True)
        else:
            scores = score_row
        scores.to_csv(outFolder + scoreFilename, index=False)
        return dict(zip(types, res_med))
    except ImportError as e:
        head = ['d{}'.format(i) for i in range(emb.shape[1])] + types
        arr = np.column_stack([emb, meta])
        arr = np.vstack([head, arr])
        path = outFolder + 'out-{}.csv'.format(name)
        saveAsCsv(arr, path)
        cmd = 'Rscript src/calculateLisi.R "{}" "{}" "{}" "{}"'.format(name,
                                                                       path,
                                                                       outFolder + scoreFilename,
                                                                       int(perplexity))
        for tp in types:
            cmd += ' \"{}\"'.format(tp)
        os.system(cmd)
        removeFile(path)
        if not fileExists(outFolder + scoreFilename):
            return None
        results = csvFile(outFolder + scoreFilename, remQuotes=True, remNewline=True)
        header = results[0][1:]
        thisResults = results[-1]
        rowName = thisResults[0]
        if rowName != name:
            print('Error: LISI file name mismatch: {} instead of {}.'.format(rowName, name))
            return None
        rowScores = thisResults[1:]
        res = {}
        for i, tp in enumerate(header):
            if rowScores[i] != 'NA':
                res[header[i]] = float(rowScores[i])
        return res


def calculateEmdMad(uncorrectedFile, correctedFile, outPath):
    """
    Calculate the Earth Mover's Distance (EMD) and Mean Absolute Deviation (MAD) between uncorrected and corrected data.

    The function computes the EMD and MAD metrics using an R script (`calculateEmdMad.R`). It evaluates the quality of
    batch correction by comparing the distribution of the uncorrected and corrected datasets.

    Parameters:
    - uncorrectedFile (str): Path to the serialized R data file containing the uncorrected data.
    - correctedFile (str): Path to the serialized R data file containing the corrected data.
    - outPath (str): Directory path where the EMD and MAD results should be saved as a CSV file.

    Returns:
    - tuple: A tuple containing:
        - float: EMD score between the uncorrected and corrected data.
        - float: MAD score between the uncorrected and corrected data.

    Note:
    The R script requires the "cyCombine" and "magrittr" libraries and makes use of the `evaluate_emd` and
    `evaluate_mad` functions from the "cyCombine" library. The corrected data is clustered using Self-Organizing Maps
    (SOM), and these clusters are used for evaluation.
    """
    cmd = 'Rscript src/calculateEmdMad.R {} {} {}'.format(uncorrectedFile, correctedFile, outPath)
    os.system(cmd)
    if not fileExists(outPath):
        return None
    results = csvFile(outPath, remQuotes=True, remNewline=True)
    vals = results[-1]
    emd = float(vals[2])
    mad = float(vals[3])
    return emd, mad


def viewHyper(uv, path=None, names:list=None):
    """
    Visualize the relationship between hyperparameter values and validation losses.

    This function generates scatter plots for each hyperparameter versus the validation loss, helping in understanding
    the sensitivity of the model's performance to various hyperparameter settings.

    Parameters:
    - uv (object): The main model or data structure object that has an `archives` attribute, specifically with 'mango'
      entries containing the optimization results.
    - path (str, optional): The directory path where the generated plots should be saved. If not provided, the plots
      will be displayed directly.
    - names (list of str, optional): A list of hyperparameter names to visualize. If not provided, all hyperparameters
      available in the 'mango' archive will be visualized.

    Returns:
    None

    Note:
    The function assumes that the 'mango' entries in the `archives` attribute of the `uv` object contain dictionaries
    with 'loss' and 'params' keys. The 'loss' key corresponds to the validation loss, and the 'params' key contains
    hyperparameter settings.
    """

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
