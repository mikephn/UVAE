import os, time, pickle, copy
from scipy.stats import randint, uniform
from src.UVAE_classes import *


hyperDefault = {'latent_dim': 50,
                'hidden': 2,
                'width': 256,
                'relu_slope': 0.2,
                'dropout': 0.0,
                'pull': 1.0,
                'lr_unsupervised': 1.0,
                'lr_supervised': 1.0,
                'lr_merge': 1.0,
                'grad_clip': 0.1,
                'ease_epochs': 1,
                'frequency': 1.0,
                'batch_size': 512
                }


hyperRanges = {'latent_dim': randint(50, 101),
               'hidden': randint(1, 3),
               'width': randint(16, 1025),
               'relu_slope': uniform(0, 0.3),
               'dropout': uniform(0, 0.3),
               'pull': uniform(0.1, 20),
               'lr_unsupervised': uniform(0.1, 1.0),
               'lr_supervised': uniform(0.1, 1.0),
               'lr_merge': uniform(0.1, 1.0),
               'grad_clip': uniform(0.0001, 1.0),
               'ease_epochs': randint(1, 10),
               'frequency': uniform(0.01, 1.0),
               'batch_size': randint(128, 1025)
               }


def optimizeHyperparameters(sourceHist, iterations, maxEpochs=30,
                            earlyStopEpochs=10,
                            samplesPerEpoch=10000,
                            valSamplesPerEpoch=5000,
                            sizeSeparately=True,
                            overwriteIfBest=False):
    sourceHist.compound()
    source = sourceHist.source
    origPath = source.path
    source.build()
    toOptimize = {}
    toOptimize.update(hyperRanges)
    allCs = source.allConstraints()
    Cs = [c for c in allCs if c.trained == False]
    if len(Cs) == 0:
        print('Nothing to optimize.')
        return
    else:
        print('Optimizing for', iterations)
    trainedAEs = [c for c in allCs if c.trained == True and type(c) is Autoencoder]
    if len(trainedAEs):
        del toOptimize['latent_dim']
    for c in Cs:
        h = c.hyperparams()
        if sizeSeparately:
            if ('hidden' in h) and ('width' in h):
                toOptimize['hidden-{}'.format(c.name)] = toOptimize['hidden']
                toOptimize['width-{}'.format(c.name)] = toOptimize['width']
        if 'frequency' in h:
            if c.weight != 0:
                toOptimize['freq-' + c.name] = toOptimize['frequency']
        if 'pull' in h and hasattr(c, 'pull'):
            toOptimize['pull-' + c.name] = toOptimize['pull']
    del toOptimize['frequency']
    del toOptimize['pull']
    if sizeSeparately:
        del toOptimize['hidden']
        del toOptimize['width']
    supervisedCs = [c for c in Cs if (type(c) is Regression) or (type(c) is Classification)]
    if len(supervisedCs) == 0:
        del toOptimize['lr_supervised']
    mergeCs = [c for c in Cs if c.trainEmbedding]
    if len(mergeCs) == 0:
        del toOptimize['lr_merge']
    easeCs = [c for c in Cs if type(c) is Normalization] + list(source.resamplings.values())
    if len(easeCs) == 0:
        del toOptimize['ease_epochs']

    def setParams(uv, params):
        hyper = dict(params)
        # only mark for training if constraint was untrained before optimization call
        for c in Cs:
            c.trained = False
            if isinstance(c, Autoencoder):
                c.encoder.trained = False
        for c in uv.allConstraints():
            if 'hidden-{}'.format(c.name) in hyper:
                c.hyper['hidden'] = hyper['hidden-{}'.format(c.name)]
                del hyper['hidden-{}'.format(c.name)]
            if 'width-{}'.format(c.name) in hyper:
                c.hyper['width'] = hyper['width-{}'.format(c.name)]
                del hyper['width-{}'.format(c.name)]
            if 'freq-{}'.format(c.name) in hyper:
                c.frequency = float(hyper['freq-{}'.format(c.name)])
                del hyper['freq-{}'.format(c.name)]
            if 'pull-{}'.format(c.name) in hyper:
                c.pull = float(hyper['pull-{}'.format(c.name)])
                del hyper['pull-{}'.format(c.name)]
        # reload = True resets untrained constraints, but loads if already trained before optimization call
        uv.build(hyper=hyper, reload=True)

    tempPath = source.path + '_opt'
    def trainFunc(params):
        if os.path.exists(tempPath):
            os.remove(tempPath)
        source.path = tempPath
        t0i = time.time()

        setParams(source, params)
        hist = source.train(maxEpochs=maxEpochs, earlyStopEpochs=earlyStopEpochs,
                        samplesPerEpoch=samplesPerEpoch, valSamplesPerEpoch=valSamplesPerEpoch,
                        saveBest=True, resetOpt=True, verbose=True)
        dt = time.time() - t0i
        print('Training time: {}s, loss: {}'.format(int(dt), hist.minValLoss))
        losses = [r['loss'] for r in sourceHist.results]
        arch = {'loss': hist.minValLoss, 'params': params, 'time': dt,
                'maxEpochs': maxEpochs, 'earlyStopEpochs': earlyStopEpochs,
                'samplesPerEpoch': samplesPerEpoch, 'valSamplesPerEpoch': valSamplesPerEpoch}
        sourceHist.addResult(arch)
        source.archive()
        best = None
        if len(losses) > 0:
            best = np.nanmin(losses)
            isBest = hist.minValLoss < best
        else:
            isBest = True
        if isBest and not np.isnan(hist.minValLoss):
            print('Improved: {} -> {}'.format(best, hist.minValLoss))
            if overwriteIfBest:
                os.replace(tempPath, origPath)
        return hist.minValLoss

    def objective(params):
        checked = []
        losses = []
        for p in params:
            msg = '\nValidating hyper-parameter set:\n'
            for k in sorted(list(p.keys())):
                msg += '\t{}: {:.2f}'.format(k, p[k])
            print(msg)
            loss = trainFunc(dict(p))
            if (not np.isnan(loss)) and (not np.isinf(loss)):
                checked.append(p)
                losses.append(loss)
        return checked, losses

    from mango import Tuner

    try:
        conf_dict = dict(num_iteration=int(iterations))
        t0 = time.time()
        tuner = Tuner(toOptimize, objective, conf_dict)
        tuner.minimize()
        t1 = time.time()
        print('Total model selection time: {}s'.format(int(t1 - t0)))
        best = tuner.results['best_params']
        print('Best set:', best)
        print('Best loss:', tuner.results['best_objective'])
        if os.path.exists(tempPath):
            os.remove(tempPath)
        source.path = origPath
        if not overwriteIfBest:
            setParams(source, best)
            sourceHist.compound()
            source.archive()
    except Exception as e:
        print(e)
