from src.tools import *
from src.UVAE import *
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# download sample data: https://zenodo.org/record/5748302/files/ToyData-3x3.pkl?download=1
toyDs = unpickle('ToyData-3x3.pkl')

# load test data
X0, X1, X2 = toyDs['Xs']
chs0, chs1, chs2 = toyDs['enum']['markers']
Y0, Y1, Y2 = toyDs['Ys']['label']
B0, B1, B2 = toyDs['Ys']['batch']

# number of repeats and epochs for each configuration
repeats = 10
n_epochs = 10

# Each config is trained and evaluated by generating three types of outputs: reconstructed (known markers),
# merged (known markers, merged from all modalities), and imputed (hidden markers obtained by cross-modal prediction).
# Pearson and Spearman correlation with ground truth are calculated for every type.
# Then min, max, and mean are obtained for each correlation across all channels, giving 6 test metrics.
# If any of the metrics is lower than the average from previous runs for the same config the test fails.
# The difference threshold for failure is specified below:
fail_diff = 0.1

# test configurations
testConfigs = [{'cond': False, 'sd': False, 'norm': False, 'mmd': False, 'resample': False, 'sub': False, 'bal_batch': True},
                {'cond': True, 'sd': False, 'norm': False, 'mmd': False, 'resample': False, 'sub': False, 'bal_batch': True},
                {'cond': True, 'sd': False, 'norm': False, 'mmd': False, 'resample': False, 'sub': True, 'bal_batch': True},
                {'cond': True, 'sd': True, 'norm': False, 'mmd': False, 'resample': True, 'sub': True, 'bal_batch': True},
                {'cond': True, 'sd': False, 'norm': True, 'mmd': False, 'resample': True, 'sub': True, 'bal_batch': True},
                {'cond': True, 'sd': False, 'norm': False, 'mmd': True, 'resample': False, 'sub': False, 'bal_batch': True},
                {'cond': True, 'sd': False, 'norm': False, 'mmd': True, 'resample': True, 'sub': False, 'bal_batch': True},
                {'cond': True, 'sd': True, 'norm': False, 'mmd': True, 'resample': True, 'sub': True, 'bal_batch': True},
                {'cond': True, 'sd': True, 'norm': True, 'mmd': True, 'resample': True, 'sub': True, 'bal_batch': True},
                {'cond': True, 'sd': True, 'norm': True, 'mmd': True, 'resample': True, 'sub': True, 'bal_batch': False},
               ]

# file with results from previous runs
baselineFile = 'baseline.pkl'

# train a model given a config
def trainConfig(config, path='test.uv', verbose=False):
    if fileExists(path):
        removeFile(path)

    uv = UVAE(path)

    p0 = uv + Data(X0, channels=chs0, name='Panel 0')
    p1 = uv + Data(X1, channels=chs1, name='Panel 1')
    p2 = uv + Data(X2, channels=chs2, name='Panel 2')

    batch = uv + Labeling(Y={p0: B0, p1: B1, p2: B2}, name='Batch')

    ctype = uv + Classification(Y={p0: Y0, p1: Y1, p2: Y2}, nullLabel='unk', name='Cell type')

    red2d = uv + Projection(latent_dim=2, name='2D')

    if config['cond']:
        conditions = [batch]
    else:
        conditions = None

    ae0 = uv + Autoencoder(name=p0.name, masks=p0, conditions=conditions)
    ae1 = uv + Autoencoder(name=p1.name, masks=p1, conditions=conditions)
    ae2 = uv + Autoencoder(name=p2.name, masks=p2, conditions=conditions)

    if config['sd']:
        sd = uv + Standardization(Y=batch.Y, name='sd', balanceBatchAvg=config['bal_batch'])
        if config['resample']:
            ctype.resample(sd)

    if config['norm']:
        norm = uv + Normalization(Y=batch.Y, name='norm', balanceBatchAvg=config['bal_batch'])
        if config['resample']:
            ctype.resample(norm)

    if config['mmd']:
        mmd = uv + MMD(Y=batch.Y, name='MMD', pull=1, frequency=3, balanceBatchAvg=config['bal_batch'])
        if config['resample']:
            ctype.resample(mmd)

    if config['sub']:
        sub = uv + Subspace(name='Sub', pull=1, conditions=conditions)

    uv.train(n_epochs, callback=None, verbose=verbose)

    return uv

# test reconstruction from a model against the ground truth values
def testCorrelations(uv):
    GT0, GT1, GT2 = toyDs['Ys']['GT_X']
    gt_chs = toyDs['enum']['GT_X']

    dm = uv.allDataMap()

    # reconstruct own markers for each panel
    rec = uv.reconstruct(dataMap=dm)
    # imputed and merged outputs (all markers)
    rec_all = uv.reconstruct(dataMap=dm, channels=gt_chs)

    imputed = {ch: {'gt': [], 'rec': []} for ch in gt_chs}
    reconstructed = {ch: {'gt': [], 'rec': []} for ch in gt_chs}
    merged = {ch: {'gt': [], 'rec': []} for ch in gt_chs}

    # sort reconstructions into the three types
    for ch in gt_chs:
        gt_xs = [GT0, GT1, GT2]
        ch_ind = list(gt_chs).index(ch)
        for pi, p_chs in enumerate([chs0, chs1, chs2]):
            gt_x = gt_xs[pi][:, ch_ind]
            rec_merged_x = rec_all[uv.data[pi]][:, ch_ind]
            if ch in p_chs:
                merged[ch]['gt'].append(gt_x)
                merged[ch]['rec'].append(rec_merged_x)

                p_ch_ind = list(p_chs).index(ch)
                rec_x = rec[uv.data[pi]][:, p_ch_ind]

                reconstructed[ch]['gt'].append(gt_x)
                reconstructed[ch]['rec'].append(rec_x)
            else:
                imputed[ch]['gt'].append(gt_x)
                imputed[ch]['rec'].append(rec_merged_x)

    def calculateCorrelations(vals):
        ch_corr = {}
        for ch in vals:
            if len(vals[ch]['gt']):
                cat_gt = np.concatenate(vals[ch]['gt'])
                cat_pred = np.concatenate(vals[ch]['rec'])
                r = pearsonr(cat_gt, cat_pred)[0]
                rho = spearmanr(cat_gt, cat_pred)[0]
                ch_corr[ch] = {'r': r, 'rho': rho}
        return ch_corr

    # calculate two types of correlation for each reconstruction type
    rec_ch_corr = calculateCorrelations(reconstructed)
    merged_ch_corr = calculateCorrelations(merged)
    imp_ch_corr = calculateCorrelations(imputed)

    return rec_ch_corr, merged_ch_corr, imp_ch_corr


def printCorrelations(rec_ch_corr, merged_ch_corr, imp_ch_corr):
    print('\nReconstructed:')
    for ch in rec_ch_corr:
        print('{}: pearson: {} spearman: {}'.format(ch, round(rec_ch_corr[ch]['r'], 4), round(rec_ch_corr[ch]['rho'], 4)))
    print('\nMerged:')
    for ch in merged_ch_corr:
        print('{}: pearson: {} spearman: {}'.format(ch, round(merged_ch_corr[ch]['r'], 4), round(merged_ch_corr[ch]['rho'], 4)))
    print('\nImputed:')
    for ch in imp_ch_corr:
        print('{}: pearson: {} spearman: {}'.format(ch, round(imp_ch_corr[ch]['r'], 4), round(imp_ch_corr[ch]['rho'], 4)))


def saveBoxplot(rec_ch_corr, merged_ch_corr, imp_ch_corr, boxplotPath):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Channel correlation')

    ax1.boxplot([[rec_ch_corr[ch]['r'] for ch in rec_ch_corr],
                [merged_ch_corr[ch]['r'] for ch in merged_ch_corr],
                [imp_ch_corr[ch]['r'] for ch in imp_ch_corr]], showmeans=False)
    ax2.boxplot([[rec_ch_corr[ch]['rho'] for ch in rec_ch_corr],
                [merged_ch_corr[ch]['rho'] for ch in merged_ch_corr],
                [imp_ch_corr[ch]['rho'] for ch in imp_ch_corr]], showmeans=False)
    ax1.set_title('Pearson')
    ax2.set_title('Spearman')
    ax1.set_xticklabels(['Reconst.', 'Merged', 'Imputed'], rotation=0)
    ax2.set_xticklabels(['Reconst.', 'Merged', 'Imputed'], rotation=0)
    if boxplotPath is not None:
        fig.savefig(boxplotPath, dpi=100, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if fileExists(baselineFile):
    baseline = unpickle(baselineFile)
else:
    baseline = {'version': 1, 'results': {}}

print('Loaded results:')
for c in baseline['results']:
    v = np.mean(baseline['results'][c])
    print(c, v)

hasFailed = False

for rep in range(repeats):
    for ii, config in enumerate(testConfigs):
        # generate a string from config to use as unique hashable key
        cf_key = ''
        for k, v in config.items():
            cf_key += '{}:{},'.format(k, v)

        print('\nTraining config ({}/{} rep. {}/{}):'.format(ii+1, len(testConfigs), rep+1, repeats))
        print(cf_key)

        uv = trainConfig(config)

        t_r, t_m, t_i = testCorrelations(uv)

        # reduce all channel results to 6 values, min, max, and mean for two correlation types
        def gather(d, k):
            return [d[ch][k] for ch in d]
        all_r = gather(t_r, 'r') + gather(t_m, 'r') + gather(t_i, 'r')
        all_rho = gather(t_r, 'rho') + gather(t_m, 'rho') + gather(t_i, 'rho')
        res = [np.min(all_r), np.mean(all_r), np.max(all_r),
               np.min(all_rho), np.mean(all_rho), np.max(all_rho)]

        # calculate the deviation from expected mean of previous runs
        if cf_key in baseline['results']:
            existingResults = baseline['results'][cf_key]
            existingMean = np.mean(existingResults, axis=0)
            existingSd = np.std(existingResults, axis=0)
            diff = existingMean - np.array(res)
            print('previous mean: {} sd: {}'.format(np.round(existingMean, 2), np.round(existingSd, 2)))
            print('now: {}'.format(np.round(res, 2)))
            print('difference: {}'.format(np.round(diff, 2)))
            if np.any(diff > fail_diff):
                hasFailed = True
                print('FAILED: {}'.format(cf_key))
            else:
                print('PASS')
            baseline['results'][cf_key].append(res)
        else:
            baseline['results'][cf_key] = [res]

    if not hasFailed:
        # if no failure, add the results and save
        doPickle(baseline, baselineFile)
    else:
        # if any failure, save as a separate file
        doPickle(baseline, baselineFile + '_failed')
