from src.UVAE_hyper import *

class UVAE:
    def __init__(self, path):
        self.data = []
        self.autoencoders = {}
        self.constraints = {}
        self.resamplings = {}
        self.hyper = {}
        self.optimizers = {}
        self.history = None
        self.msHistory = ModelSelectionHistory(self)
        self.archives = {'autoencoders': {},
                         'constraints': {},
                         'resamplings': {},
                         'histories': []}
        self.path = path
        self.built = False
        self.variational = True
        self.shouldCache = False
        if os.path.exists(path):
            self.unarchive()

    def train(self, maxEpochs=30, batchSize=None, samplesPerEpoch=0, valSamplesPerEpoch=0, earlyStopEpochs=0, callback=None, saveBest=False, resetOpt=True, skipTrained=True, verbose=True):
        self.build(overwrite=not skipTrained)
        if batchSize is not None:
            self.hyper['batch_size'] = int(batchSize)
        if verbose:
            print('Training UVAE in:', self.path)
            print('Data:')
            print(['{}: {}'.format(d.name, int(len(d.X))) for d in self.data])
            print('Trainable constraints:')
            print(list([c for c in self.constraints if self.constraints[c].trained == False]))
            print('Static constraints:')
            print(list([c for c in self.constraints if self.constraints[c].trained == True]))
        if (len(self.optimizers) == 0) or resetOpt:
            self.optimizers['unsupervised'] = keras.optimizers.Adam(0.001*float(self.hyperparams()['lr_unsupervised']))
            self.optimizers['supervised'] = keras.optimizers.Adam(0.001*float(self.hyperparams()['lr_supervised']))
            self.optimizers['merge'] = keras.optimizers.Adam(0.001*float(self.hyperparams()['lr_merge']))
        if valSamplesPerEpoch > 0:
            self.history = History(earlyStop=earlyStopEpochs, earlyStopKey='val_loss')
        else:
            self.history = History(earlyStop=earlyStopEpochs, earlyStopKey='loss')

        for ep in range(maxEpochs):
            self.history.time('epoch')
            h = self.propagate(validation=False, batchSize=int(self.hyperparams()['batch_size']),
                               sampleLimit=samplesPerEpoch, skipTrained=skipTrained)
            if h is None:
                print('Nothing to train.')
                return self.history
            ease_prop = min(1.0, ep / max(1.0, float(self.hyperparams()['ease_epochs'])))
            self.resample(resample_prop=ease_prop, skipTrained=skipTrained)
            self.updateOffsets(correction_prop=ease_prop, skipTrained=skipTrained)
            if valSamplesPerEpoch > 0:
                self.propagate(validation=True, batchSize=self.hyperparams()['batch_size'],
                               sampleLimit=valSamplesPerEpoch, skipTrained=skipTrained)
            self.history.accumulate(sum=False)
            if verbose:
                self.history.print()
            if callback is not None:
                try:
                    callback(self)
                except Exception as ex:
                    print('Callback error.', ex)
            if np.isnan(self.history.history['loss'][-1]):
                break
            if saveBest:
                if self.history.improved:
                    self.archive()
            else:
                self.archive()
            if self.history.shouldStop:
                break
        for c in self.allConstraints():
            c.trained = True
            if isinstance(c, Autoencoder):
                c.encoder.trained = True
        self.loadParams()
        self.archives['histories'].append(self.history.history)
        self.archive()
        return self.history

    def optimize(self, iterations=20, maxEpochs=30, earlyStopEpochs=0,
                 samplesPerEpoch=100000, valSamplesPerEpoch=100000,
                 subset=None, sizeSeparately=False, lossWeight=1.0,
                 lisiValidationSet=None, customLoss=None, callback=None):
        left = self.msHistory.addIterations(iterations)
        if left > 0:
            optimizeHyperparameters(self.msHistory, iterations=left, maxEpochs=maxEpochs, earlyStopEpochs=earlyStopEpochs,
                                    samplesPerEpoch=samplesPerEpoch, valSamplesPerEpoch=valSamplesPerEpoch,
                                    subset=subset, sizeSeparately=sizeSeparately, lossWeight=lossWeight,
                                    lisiValidationSet=lisiValidationSet, customLoss=customLoss, callback=callback)

    def archive(self, path=None):
        if not self.built:
            self.build()
        if path is None:
            path = self.path
            a = self.archives
        else:
            a = dict(self.archives)
        a['hyper'] = self.hyper
        a['msel'] = {'results': self.msHistory.currentResults,
                     'pastResults': self.msHistory.pastResults}
        for ae in self.autoencoders.values():
            a['autoencoders'][ae.name] = ae.archive()
        for c in self.constraints.values():
            a['constraints'][c.name] = c.archive()
        for c, targets in self.resamplings.items():
            a['resamplings'][c.name] = [t.name for t in targets]
        pickle.dump(a, open(path, "wb"))

    def unarchive(self):
        d = pickle.load(open(self.path, "rb"))
        self.archives = d
        self.hyper = d['hyper']
        self.msHistory.currentResults = d['msel']['results']
        self.msHistory.pastResults = d['msel']['pastResults']
        for ae_name, ae_arch in d['autoencoders'].items():
            ae = Autoencoder(ae_name, variational=ae_arch['variational'])
            ae.parent = self
            ae.unarchive(ae_arch)
            for dname, channels in ae_arch['input'].items():
                x = np.zeros((0, len(channels)), dtype=float)
                data = self + Data(X=x, name=dname, channels=channels)
                ae.addMask(data, np.zeros(0))
                self.autoencoders[data] = ae
        for c_name, c_arch in d['constraints'].items():
            c_class = getattr(sys.modules[__name__], c_arch['class'])
            if issubclass(c_class, Autoencoder):
                c = c_class(name=c_name, variational=c_arch['variational'])
            else:
                c = c_class(name=c_name)
            c.parent = self
            c.unarchive(c_arch)
            self.constraints[c_name] = c
            if 'input' in c_arch:
                for d_name in c_arch['input']:
                    c.addMask(self[d_name], np.zeros(0))
        for c_name, targets in d['resamplings'].items():
            c = self[c_name]
            if isinstance(c, Classification):
                for t_name in targets:
                    t = self[t_name]
                    if isinstance(t, Control):
                        c.resample(t)
        for c in list(self.autoencoders.values()) + list(self.constraints.values()):
            if isinstance(c, Autoencoder):
                if 'conditions' in c.saved:
                    c.conditions = [self[c_name] for c_name in c.saved['conditions']]

    def loadParams(self):
        for c in self.allConstraints():
            if c.trained:
                c.loadParams()

    def hyperparams(self):
        hyper = dict(hyperDefault)
        hyper.update(self.hyper)
        return hyper

    def build(self, hyper=None, overwrite=False, reload=False):
        if hyper is not None:
            self.hyper.update(hyper)
        hyper = self.hyperparams()
        self.latent_dim = int(hyper['latent_dim'])
        for data in self.data:
            if data not in self.autoencoders:
                ae = Autoencoder(data.name, variational=self.variational)
            else:
                ae = self.autoencoders[data]
            self.autoencoders[data] = ae
            ae.parent = self
            ae.in_dim = len(data.channels)
            ae.addMask(data, np.ones(len(data.X)))
            if (ae.func is None) or overwrite or reload:
                ae.latent_dim = self.latent_dim
                ae.build()

        encoders = {data: self.autoencoders[data].encoder for data in self.autoencoders}

        for ae in self.autoencoders.values():
            ae.decoder.embedding = encoders

        for reg in self.constraintsType(Classification) + self.constraintsType(Regression):
            reg.embedding = encoders
            if (reg.func is None) or overwrite or reload:
                reg.in_dim = self.latent_dim
                reg.build()

        for sub in self.constraintsType(Subspace):
            sub.embedding = encoders
            if (sub.func is None) or overwrite or reload:
                sub.latent_dim = self.latent_dim
                sub.build()

        for md in self.constraintsType(MMD):
            md.embedding = encoders
            md.index()

        for b in self.constraintsType(Normalization) + self.constraintsType(Standardization) + self.constraintsType(Labeling):
            b.index()

        for proj in self.constraintsType(Projection):
            proj.encoder.embedding = encoders
            if (proj.func is None) or overwrite or reload:
                proj.in_dim = self.latent_dim
                proj.build()

        if not overwrite:
            self.loadParams()
        self.built = True
        self.hyper = hyper

    def propagate(self, validation=False, batchSize=128, sampleLimit=0, skipTrained=True):
        # get constraints trainable by backprop
        constraints = self.allConstraints()
        constraints = [c for c in constraints if c._inds is not None and type(c) not in [Standardization, Normalization, Labeling]]
        if skipTrained:
           constraints = [c for c in constraints if not c.trained]
        # balance training labels where necessary
        for c in constraints:
            if type(c) is Classification and c.equalizeLabels:
                c.balance(None)

        # determine number of batches in each constraint
        n_batches = [c.length(batchSize, validation=validation) for c in constraints]

        if np.sum(n_batches) == 0:
            if validation:
                print('No validation samples. Please set valProp>0 when adding Data constraints.')
            return None
        # scale number of batches in training
        if not validation:
            for cn, const in enumerate(constraints):
                if const.frequency != 1.0:
                    n_batches[cn] = int(np.round(n_batches[cn] * const.frequency))
        # define current batch index and order of training proportional to number of batches
        b_order = np.concatenate([np.repeat(n, n_batches[n]) for n in range(len(constraints))])
        np.random.shuffle(b_order)
        if sampleLimit > 0:
            lim_b = max(1, int(sampleLimit/batchSize))
            b_order = b_order[:lim_b]
        # to accumulate constraint loss * constraint weight
        weighed_losses = {c: [] for c in constraints}
        if UVAE_DEBUG:
            print('Training epoch: {}'.format(self.history.epoch))

        for b_id, c_n in enumerate(b_order):
            # get the constraint to be trained in this batch
            const = constraints[c_n]
            # get batch input data for the constraint
            inds = const.batch(batchSize, validation=validation)

            # calculate loss tensors for the batch
            with tf.GradientTape(persistent=True) as tape:
                losses, w_loss = const.forward(inds)
                weighed_losses[const].append(w_loss)
                if const.adversarial and const.trainEmbedding:
                    losses[const.name+'-merge'] = -losses[const.name]

            def gradClip(grads:list, weights:list):
                gs = []
                ws = []
                for i, g in enumerate(grads):
                    if g is not None:
                        gs.append(tf.clip_by_norm(g, float(self.hyperparams()['grad_clip'])))
                        ws.append(weights[i])
                return gs, ws

            # backpropagate if in training
            if not validation:
                if const.func is not None and const.name in losses:
                    loss = losses[const.name]
                    weights = []
                    weights.extend(const.func.trainable_weights)
                    grads = tape.gradient(loss, weights)
                    grads, weights = gradClip(grads, weights)
                    const.invalidateCache()

                    if isinstance(const, Autoencoder):
                        self.optimizers['unsupervised'].apply_gradients(zip(grads, weights))
                        const.encoder.invalidateCache()
                        if UVAE_DEBUG > 1:
                            print('Training AE:', const)

                    if (type(const) is Regression) or (type(const) is Classification):
                        self.optimizers['supervised'].apply_gradients(zip(grads, weights))
                        if UVAE_DEBUG > 1:
                            print('Training:', const)

                if const.trainEmbedding and len(const.embedding):
                    if const.name+'-merge' in losses:
                        emb_loss = losses[const.name + '-merge']
                    else:
                        emb_loss = losses[const.name]
                    embedding_weights = []
                    for data in const.embedding:
                        encoder = const.embedding[data]
                        if (not encoder.trained) or (not skipTrained):
                            embedding_weights.extend(encoder.func.trainable_weights)
                            encoder.invalidateCache()
                            if UVAE_DEBUG > 1:
                                print('Training embedding (from {}):'.format(const.name), encoder.name)
                    if type(const) is Subspace:
                        embedding_weights.extend(const.encoder.func.trainable_weights)
                    grads = tape.gradient(emb_loss, embedding_weights)
                    grads, embedding_weights = gradClip(grads, embedding_weights)
                    self.optimizers['merge'].apply_gradients(zip(grads, embedding_weights))

            # save losses
            for k in losses:
                if not validation:
                    self.history.append(k, losses[k])
                else:
                    self.history.append('val_'+k, losses[k])

        weighed_loss = 0
        n_c = 0
        for c in weighed_losses:
            if not c.adversarial and len(weighed_losses[c]):
                weighed_loss += np.nanmean(weighed_losses[c])
                n_c += 1
        weighed_loss /= n_c

        if validation:
            self.history.append('val_loss', weighed_loss)
        else:
            self.history.epoch += 1
            self.history.append('loss', weighed_loss)

        return self.history

    def resample(self, resample_prop=1.0, skipTrained=True):
        if len(self.resamplings):
            # predict merged targets for each classifier
            targetMaps = {}
            targetResults = {}
            for clsf, targets in self.resamplings.items():
                combinedMap = {}
                for target in targets:
                    if not target.trained or not skipTrained:
                        t_inds = target.inds(validation=False, controls=True, resampled=False)
                        d_map = target.dataMap(t_inds)
                        targetMaps[target] = d_map
                        for data, inds in d_map.items():
                            if data not in combinedMap:
                                combinedMap[data] = inds
                            else:
                                combinedMap[data] = np.sort(np.unique(np.concatenate([combinedMap[data], inds])))
                if len(combinedMap) == 0:
                    continue
                if UVAE_DEBUG:
                    print('Predicting ({}): '.format(clsf.name), [len(combinedMap[d]) for d in combinedMap])
                called = clsf.predictMap(combinedMap)
                ref = Classification(None)
                for data in called:
                    mask = np.zeros(len(data.X), dtype=bool)
                    mask[np.array(combinedMap[data], dtype=int)] = True
                    ref.addMask(data, mask)
                    ref.setTargets(data, called[data])
                ref.index()
                for target in targets:
                    if not target.trained or not skipTrained:
                        if UVAE_DEBUG:
                            print('Resampling:', target.name)
                        dmap = targetMaps[target]
                        t_pred = ref.YsFromMap(dmap)
                        if target not in targetResults:
                            targetResults[target] = [t_pred]
                        else:
                            targetResults[target].append(t_pred)
            for target in targetResults:
                target.balance(targetResults[target], prop=resample_prop)

    def updateOffsets(self, correction_prop=1.0, skipTrained=True):
        norm_cs = self.constraintsType(Normalization) + self.constraintsType(Standardization)
        for c in norm_cs:
            if not c.trained or not skipTrained:
                c.intervalsLeft -= 1
                if c.intervalsLeft == 0:
                    c.intervalsLeft = c.interval
                    if UVAE_DEBUG:
                        print('Updating offsets:', c.name)
                    if type(c) is Normalization:
                        # for data autoencoders
                        encoders = {data: self.autoencoders[data].encoder for data in
                                    self.autoencoders if type(self.autoencoders[data].encoder) is Unbiasing
                                    and data in c.targets}
                        c.calculateBias(encoders, prop=correction_prop)
                        # for subspace autoencoders
                        for sub in self.constraintsType(Subspace):
                            if type(sub.encoder) is Unbiasing:
                                c.calculateBias({data: sub.encoder for data in sub.masks}, prop=correction_prop)
                    elif type(c) is Standardization:
                        c.calculateStats()
                        c.standardizeData()

    def predictMap(self, dataMap, mean=False, stacked=False, bs=4096):
        if not self.built:
            self.build()
        emb = {}
        for data in dataMap:
            emb[data] = self.autoencoders[data].encoder.predictMap({data: dataMap[data]}, mean=mean, bs=bs)[data]
        if stacked:
            emb = np.array(np.concatenate(list(emb.values())))
        return emb

    def mergedPredictMap(self, dataMap, embeddings:list=None, uniform=True, prop=0.5, stacked=False):
        embs_own = self.predictMap(dataMap, mean=True)
        if embeddings is None or len(embeddings) == 0:
            return embs_own
        embs = {dt: [[samp] for samp in embs_own[dt]] for dt in embs_own}
        for e in embeddings:
            e_map = {dt: np.arange(len(e.masks[dt]))[e.masks[dt]] for dt in e.masks}
            for dt in e_map:
                common, ind_own, _ = np.intersect1d(dataMap[dt], e_map[dt], return_indices=True)
                emb = e.predictMap({dt: common}, mean=True)[dt]
                for ii, i in enumerate(ind_own):
                    embs[dt][i].append(emb[ii])
        for dt in embs:
            if not uniform:
                if len(embs[dt][0]) == 2:
                    means = []
                    for c in embs[dt]:
                        cm = (c[0] * prop) + (c[1] * (1.0-prop))
                        means.append(cm)
                    embs[dt] = np.array(means)
                else:
                    embs[dt] = np.array([np.mean(c, axis=0) for c in embs[dt]])
            else:
                for levels in range(1, len(embeddings) + 2):
                    level_inds = [i for i in range(len(embs[dt])) if len(embs[dt][i]) == levels]
                    level_samps = [embs[dt][i] for i in level_inds]
                    if len(level_samps):
                        if levels == 1:
                            for ii, i in enumerate(level_inds):
                                embs[dt][i] = level_samps[ii][0]
                        else:
                            r = np.random.exponential(1, size=(len(level_samps), levels))
                            r_normed = r / np.sum(r, axis=1)[..., None]
                            for ii, i in enumerate(level_inds):
                                props = r_normed[ii]
                                weighed = [level_samps[ii][ei] * props[ei] for ei in range(levels)]
                                embs[dt][i] = np.sum(weighed, axis=0)
            embs[dt] = np.array(embs[dt])
        if stacked:
            return np.array(np.concatenate(list(embs.values())))
        return embs

    def reconstruct(self, dataMap, channels=None, bs=1024, stacked=False, mean=True):
        rec = {}
        for d in dataMap:
            p_dm = {d: dataMap[d]}
            if channels is None:
                rec[d] = self.autoencoders[d].decoder.predictMap(p_dm, mean=mean, bs=bs)[d]
            else:
                useDecoders = {}
                for data in self.autoencoders:
                    if len(set(channels).intersection(set(data.channels))):
                        useDecoders[data] = self.autoencoders[data].decoder
                if not len(useDecoders):
                    print('Error: no decoders contain specified channels.')
                    return None
                accum = {ch: [] for ch in channels}
                for data, dec in useDecoders.items():
                    r = dec.predictMap(p_dm, mean=mean, bs=bs)[d]
                    for ci, ch in enumerate(data.channels):
                        if ch in accum:
                            accum[ch].append(r[:, ci])
                means = [np.mean(accum[ch], axis=0) for ch in accum]
                rec[d] = np.transpose(np.array(means))
        if stacked:
            return np.array(np.concatenate(list(rec.values())))
        return rec


    def addConstraint(self, const):
        if type(const) is Data:
            const = self.appendData(const)
        else:
            const = self.appendConstraint(const)
        if const is not None:
            const.parent = self
        return const

    def removeConstraint(self, const):
        if const.name in self.constraints:
            if const in self.data:
                self.data.remove(const)
            if const in self.constraints.values():
                del self.constraints[const.name]
            if const in self.resamplings:
                del self.resamplings[const]
            const.parent = None
            if const.name in self.archives['autoencoders']:
                del self.archives['autoencoders'][const.name]
            if const.name in self.archives['constraints']:
                del self.archives['constraints'][const.name]
            if const.name in self.archives['resamplings']:
                del self.archives['resamplings'][const.name]
            return self

    def allDataMap(self, subsample=0):
        dm = DataMap()
        sum = np.sum([len(data.X) for data in self.data])
        for data in self.data:
            if len(data.X):
                if subsample == 0:
                    inds = np.arange(len(data.X))
                else:
                    prop = float(subsample) / sum
                    n = int(np.ceil(len(data.X) * prop))
                    inds = np.random.permutation(len(data.X))[:n]
                dm[data] = inds
        return dm

    def constraintsType(self, t):
        return [c for c in self.constraints.values() if type(c) is t]

    def allConstraints(self):
        return list(self.constraints.values()) + list(self.autoencoders.values())

    def __getitem__(self, name):
        for d in self.data:
            if d.name == name:
                return d
        if name in self.constraints:
            return self.constraints[name]
        return None

    def __add__(self, const):
        return self.addConstraint(const)

    def __iadd__(self, const):
        self + const
        return self

    def __sub__(self, const):
        return self.removeConstraint(const)

    def __isub__(self, const):
        self - const
        return self

    def uniqueName(self, base, names):
        name = base
        i = 0
        while name in names:
            i += 1
            name = '{}_({})'.format(base, int(i))
        return name

    def appendData(self, const):
        existingNames = [d.name for d in self.data]
        if const.name is None:
            const.name = self.uniqueName('Data', existingNames)
        else:
            const.name = const.name.replace(" ", "_")
        if const.channels is None:
            const.channels = ['{} ch.{}'.format(const.name, cn) for cn in range(const.X.shape[-1])]
        if const.name in existingNames:
            existing = [d for d in self.data if d.name == const.name][0]
            if np.all(const.channels == existing.channels):
                existing.X = np.vstack((existing.X, const.X))
                existing.valProp = const.valProp
                existing.defineValMask()
                const = existing
            else:
                print(
                    'Incompatible channels for {}.\nGiven:\n{}\nExpected:\n{}'.format(const.name, const.channels,
                                                                                       existing.channels))
                return None
        else:
            self.data.append(const)
            const.defineValMask()
        const.parent = self
        return const

    def appendConstraint(self, const):
        if type(const) is Autoencoder and len(const.masks) == 1:
            data = list(const.masks.keys())[0]
            if data in self.autoencoders:
                const = self.autoencoders[data]
            const.name = data.name
            self.autoencoders[data] = const
            return const

        if const.name is None:
            const.name = self.uniqueName(type(const).__name__, list(self.constraints.keys()))
        if len(const.masks) == 0:
            # data reference was unspecified, assume reference to previously added
            if (len(self.data) == 0) and (type(const.Y) is not dict):
                print('Add Data before specifying constraints.')
                return None
            if (isinstance(const, Projection)) or (type(const) is Subspace):
                # cover all added Data by default
                for d in self.data:
                    const.addMask(d, np.ones(len(d.X), dtype=bool))
            else:
                # Regression constraint
                if const.Y is not None:
                    if type(const.Y) is not dict:
                        # Data mapping not explicitly specified, find last added Data matching sample dimension
                        for d in reversed(self.data):
                            if len(d.X) == len(const.Y):
                                const.Y = {d: np.array(const.Y)}
                                break
                        if type(const.Y) is not dict:
                            print('{} did not match any Data in sample dimension ({}).'.format(const, int(len(const.Y))))
                            return None
                    for data, targets in const.Y.items():
                        targets = np.array(targets)
                        if (isinstance(const, Classification) == False) and len(targets.shape) == 1:
                            targets = np.expand_dims(targets, -1)
                        if len(data.X) != len(targets):
                            print(
                                'Incompatible shape between {} and {}. Targets must be the same length as X for given data. Use value specified as nullLabel to fill unknown entries.'.format(data, const))
                            return None
                        if const.nullLabel is None:
                            mask = np.ones(len(targets), dtype=bool)
                        else:
                            if type(const.nullLabel) is np.floating and np.isnan(const.nullLabel):
                                mask = np.isnan(np.array(targets))
                            else:
                                mask = targets != const.nullLabel
                            if len(mask) == 0:
                                print('')
                        const.addMask(data, mask)
                        const.setTargets(data, targets[mask])

        if const.name in self.constraints:
            existing = self.constraints[const.name]
            existing.masks.update(const.masks)
            if hasattr(const, 'Y'):
                existing.Y.update(const.Y)
            if hasattr(const, 'targets'):
                existing.targets.update(const.targets)
            existing.controlMasks.update(const.controlMasks)
            const = existing
        else:
            self.constraints[const.name] = const
        const.parent = self
        return const
