import sys, os, time, pickle, copy
from collections.abc import Callable
from scipy.stats import randint, uniform
from src.UVAE_classes import *
from src.UVAE_diag import calculateLISI, classNormalizationMask

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
               'frequency': uniform(0.1, 3.0),
               'batch_size': randint(128, 1025)
               }

def lisiScoring(model, lisiValidationSet, loss, archDict, lossMsg):
    lisiValidationSet.update()
    emb = model.predictMap(lisiValidationSet.dm, mean=True, stacked=True)
    batch = lisiValidationSet.batch
    labels = lisiValidationSet.labeling
    if lisiValidationSet.normClasses:
        if not hasattr(lisiValidationSet, 'normMask'):
            lisiValidationSet.normMask = classNormalizationMask(batches=batch,
                                                                labels=labels)
        emb = emb[lisiValidationSet.normMask]
        batch = batch[lisiValidationSet.normMask]
        labels = labels[lisiValidationSet.normMask]

    lisiScores = calculateLISI(emb=emb, batches=batch,
                               classes=labels,
                               perplexity=lisiValidationSet.perplexity,
                               name='hyperopt',
                               outFolder=os.path.dirname(model.path) + '/',
                               scoreFilename='hyper_lisi.csv')
    archDict.update(lisiScores)
    archDict.update({'lisiBatchRange': lisiValidationSet.batchRange,
                 'lisiLabelRange': lisiValidationSet.labelRange,
                 'lisiBatchWeight': lisiValidationSet.batchWeight,
                 'lisiLabelWeight': lisiValidationSet.labelWeight})

    def weighedRangeScore(score, bottom, top, weight):
        clipped = max(bottom, min(top, score))
        scaled = (clipped - bottom) / (top - bottom)
        return scaled * weight

    batchScore = weighedRangeScore(score=lisiScores['batch'],
                                   bottom=lisiValidationSet.batchRange[0],
                                   top=lisiValidationSet.batchRange[1],
                                   weight=lisiValidationSet.batchWeight)  # higher the better
    labelScore = weighedRangeScore(score=lisiScores['class'],
                                   bottom=lisiValidationSet.labelRange[0],
                                   top=lisiValidationSet.labelRange[1],
                                   weight=lisiValidationSet.labelWeight)  # lower the better
    loss -= (batchScore - labelScore)  # subtract for minimisation objective
    lossMsg += '\n- LISI batch: {}\n+ LISI class: {}'.format(batchScore, labelScore)
    return loss, archDict, lossMsg


def optimizeHyperparameters(sourceHist, iterations, maxEpochs=30,
                            earlyStopEpochs=10,
                            samplesPerEpoch=10000,
                            valSamplesPerEpoch=5000,
                            subset=None,
                            sizeSeparately=False,
                            lossWeight=1.0,
                            lisiValidationSet:LisiValidationSet=None,
                            customLoss:Callable[..., float]=None,
                            callback=None):
    sourceHist.compound()
    source = sourceHist.source
    source.build()
    toOptimize = {}
    toOptimize.update(hyperRanges)

    allCs = source.allConstraints()
    Cs = [c for c in allCs if c.trained == False]
    if len(Cs) == 0:
        print('No untrained constraints to optimize.')
        return

    c_expanded_params = {}
    for c in Cs:
        h = list(c.hyperparams().keys())
        h = [k for k in h if k in toOptimize]
        if 'frequency' in h:
            if c.weight != 0:
                c_expanded_params['frequency-' + c.name] = toOptimize['frequency']
        if 'pull' in h and hasattr(c, 'pull'):
            c_expanded_params['pull-' + c.name] = toOptimize['pull']
        if sizeSeparately:
            if ('hidden' in h) and ('width' in h):
                c_expanded_params['hidden-{}'.format(c.name)] = toOptimize['hidden']
                c_expanded_params['width-{}'.format(c.name)] = toOptimize['width']

    unusedParams = set()
    unusedParams.add('frequency')
    unusedParams.add('pull')
    if sizeSeparately:
        unusedParams.add('hidden')
        unusedParams.add('width')
    trainedAEs = [c for c in allCs if c.trained == True and type(c) is Autoencoder]
    if len(trainedAEs):
        unusedParams.add('latent_dim')
    supervisedCs = [c for c in Cs if (type(c) is Regression) or (type(c) is Classification)]
    if len(supervisedCs) == 0:
        unusedParams.add('lr_supervised')
    mergeCs = [c for c in Cs if c.trainEmbedding]
    if len(mergeCs) == 0:
        unusedParams.add('lr_merge')
    easeCs = [c for c in Cs if type(c) is Normalization] + list(source.resamplings.values())
    if len(easeCs) == 0:
        unusedParams.add('ease_epochs')

    toOptimize.update(c_expanded_params)
    for k in unusedParams:
        if k in toOptimize:
            del toOptimize[k]

    if subset is not None:
        keep = {}
        for kk in subset:
            for k in toOptimize:
                if k.startswith(kk):
                    keep[k] = toOptimize[k]
        toOptimize = keep

    if len(toOptimize) == 0:
        print('No hyper-parameters to optimize.')
        return
    else:
        print('Optimizing for', iterations)

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
            if 'frequency-{}'.format(c.name) in hyper:
                c.frequency = float(hyper['frequency-{}'.format(c.name)])
                del hyper['frequency-{}'.format(c.name)]
            if 'pull-{}'.format(c.name) in hyper:
                c.pull = float(hyper['pull-{}'.format(c.name)])
                del hyper['pull-{}'.format(c.name)]
        # reload = True resets untrained constraints, but loads if already trained before optimization call
        uv.build(hyper=hyper, reload=True)

    def trainFunc(params):
        t0i = time.time()

        setParams(source, params)
        hist = source.train(maxEpochs=maxEpochs, earlyStopEpochs=earlyStopEpochs,
                        samplesPerEpoch=samplesPerEpoch, valSamplesPerEpoch=valSamplesPerEpoch,
                        saveBest=True, resetOpt=True, verbose=True)
        dt = time.time() - t0i
        print('Training time: {}s, loss: {}'.format(int(dt), hist.minValLoss))
        arch = {'minLoss': hist.minValLoss, 'hist': hist.history, 'params': params, 'time': dt,
                'maxEpochs': maxEpochs, 'earlyStopEpochs': earlyStopEpochs, 'lossWeight': lossWeight,
                'samplesPerEpoch': samplesPerEpoch, 'valSamplesPerEpoch': valSamplesPerEpoch}
        if earlyStopEpochs > 0:
            loss = hist.minValLoss # lowest value of loss
        else:
            loss = hist.history[hist.earlyStopKey][-1] # last value of loss
        lossMsg = 'Train/val loss: {} * {}'.format(loss, lossWeight)
        loss *= lossWeight
        if lisiValidationSet is not None:
            loss, archDict, lossMsg = lisiScoring(source, lisiValidationSet, loss, arch, lossMsg)
        if customLoss is not None:
            loss += customLoss(source)
        lossMsg += '\n= {}'.format(loss)
        print(lossMsg)
        arch['loss'] = loss
        previousLosses = [r['loss'] for r in sourceHist.currentResults]
        sourceHist.addResult(arch)
        source.archive()
        best = None
        if len(previousLosses) > 0:
            best = np.nanmin(previousLosses)
            isBest = loss < best
        else:
            isBest = True
        if isBest and not np.isnan(loss):
            print('Improved: {} -> {}'.format(best, loss))
        if callback is not None:
            try:
                callback(source)
            except Exception as e:
             print('Optimization callback exception:', e)
        return loss

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

    # try:
    conf_dict = dict(num_iteration=int(iterations))
    t0 = time.time()
    tuner = Tuner(toOptimize, objective, conf_dict)
    tuner.minimize()
    t1 = time.time()
    print('Total model selection time: {}s'.format(int(t1 - t0)))
    best = tuner.results['best_params']
    print('Best set:', best)
    print('Best loss:', tuner.results['best_objective'])
    sourceHist.compound()
    setParams(source, best)
    source.archive()
    # except Exception as e:
    #     print('Optimization exception:', e)
