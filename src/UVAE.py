from src.UVAE_hyper import *


class UVAE:
    """
    A Variational Autoencoder with a shared latent space.

    The UVAE class provides functionality to integrate disjoint data modalities, correct
    batch effects, perform regression, classification, and imputation over
    the joint latent space. Data, labelling, and other constraints can be added
    to an instance of this class. The state of the UVAE object is automatically saved
    during training, and can be loaded automatically by specifying a file path upon initialization.

    Attributes
    ----------
    data : list
        List of added Data instances, in the order they were added.
    autoencoders : dict
        Dictionary of Autoencoder instances corresponding to each Data object.
    constraints : dict
        Dictionary containing non-autoencoder constraints. The keys are the names of the constraints.
    resamplings : dict
        Dictionary mapping resampling sources (of type Classification or Labeling) to sets of targets.
    hyper : dict
        Dictionary of shared hyper-parameters of the model.
    optimizers : dict
        Dictionary of Keras optimizers used for training.
    history : History, optional
        An instance of the History class used for early stopping and storing training losses.
    msHistory : ModelSelectionHistory
        An instance storing scores of hyper-parameter optimization using the mango library.
    archives : dict
        Dictionary of archived parameters for all model constraints.
    path : str
        File path for archiving and unarchiving the UVAE model.
    built : bool
        Indicator of whether the model constraints were instantiated and parameters were loaded.
    variational : bool
        If True, the model uses variational auto-encoders.
    shouldCache : bool
        If True, predictions and latent embeddings from constraints are cached. They are regenerated only if
        the hierarchy of the corresponding constraint changes. If False, new predictions are always made.
    """
    def __init__(self, path: str):
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
        """
        Train the UVAE model.

        Parameters
        ----------
        maxEpochs : int, optional
            Maximum number of epochs for training, by default 30.
        batchSize : int, optional
            Size of the training batch, if None the value from hyper-parameter dictionary is used.
        samplesPerEpoch : int, optional
            Number of samples per epoch, setting of 0 will pass through all added samples as one epoch.
        valSamplesPerEpoch : int, optional
            Number of validation samples per epoch, setting of 0 uses all validation data.
        earlyStopEpochs : int, optional
            Number of epochs without improvement for triggering early stopping, 0 disables early stopping.
        callback : function, optional
            Function to be called after each epoch of training.
        saveBest : bool, optional
            Whether to save the model during training after every improvement of loss, by default False.
        resetOpt : bool, optional
            Whether to reset the optimizers state before training, by default True.
        skipTrained : bool, optional
            Whether to skip training of the already trained constraints (freeze them), by default True.
        verbose : bool, optional
            Whether to print training logs, by default True.

        Returns
        -------
        History
            History object containing loss values during training.
        """
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
        """
        Optimize the UVAE model's hyper-parameters.
        Only new constraints which were not already trained are optimised (their .trained property is False).

        Parameters
        ----------
        iterations : int, optional
            Number of training attempts, by default 20.
        maxEpochs : int, optional
            Maximum number of epochs for training during each optimization run, by default 30.
        earlyStopEpochs : int, optional
            Number of epochs for early stopping, by default 0 (no early stopping).
        samplesPerEpoch : int, optional
            Number of data samples considered per epoch, by default 100000. Set 0 to pass through all the data in each epoch.
        valSamplesPerEpoch : int, optional
            Number of validation samples considered per epoch, by default 100000. This subset is fixed throughout training. Set 0 to use all validation data once.
        subset : list, optional
            List of hyper-parameter names to be optimized. To include pull or frequency of individual constraints, use pull-name or frequency-name with the constraint name. By default None.
        sizeSeparately : bool, optional
            If True, optimizes width and depth of each constraint separately, by default False.
        lossWeight : float, optional
            Adjusts the importance of the training loss for model selection (compared to LISI losses or optional custom loss), by default 1.0.
        lisiValidationSet : LisiValidationSet, optional
            Object storing information for LISI calculation, by default None.
        customLoss : function, optional
            Custom function to compute a loss contribution after each optimization run, by default None.
        callback : function, optional
            Function to call after each optimization run, by default None.

        Returns
        -------
        None
        """
        left = self.msHistory.addIterations(iterations)
        if left > 0:
            optimizeHyperparameters(self.msHistory, iterations=left, maxEpochs=maxEpochs, earlyStopEpochs=earlyStopEpochs,
                                    samplesPerEpoch=samplesPerEpoch, valSamplesPerEpoch=valSamplesPerEpoch,
                                    subset=subset, sizeSeparately=sizeSeparately, lossWeight=lossWeight,
                                    lisiValidationSet=lisiValidationSet, customLoss=customLoss, callback=callback)


    def archive(self, path=None):
        """
        Archive the UVAE model's current state, including hyperparameters, autoencoders, constraints, and resamplings.

        This method saves the current state of the UVAE model into a pickle file, which can be used later for restoration
        or analysis. If the model has not been built, it triggers the build process before archiving.

        Parameters
        ----------
        path : str, optional
            The path to save the pickle archive. If not provided, uses the default path set during the model's initialization.

        Returns
        -------
        None
        """
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
        """
        Restore the UVAE model's state from a previously saved archive.

        This method loads the UVAE model's state from a pickle file saved on the default path set during the
        model's initialization. It restores hyperparameters, autoencoders, constraints, resamplings, and other
        related attributes to their saved states.

        Returns
        -------
        None
        """
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
        """
        Loads the saved parameters of each of model constraints.

        Returns
        -------
        None
        """
        for c in self.allConstraints():
            if c.trained:
                c.loadParams()


    def hyperparams(self):
        """
        Retrieve the hyperparameters of the UVAE model.

        This method returns a dictionary of hyperparameters for the UVAE model. If any custom hyperparameters
        have been set, they will override the default values in the returned dictionary.

        Returns
        -------
        dict
            Dictionary containing the hyperparameters of the UVAE model.
        """
        hyper = dict(hyperDefault)
        hyper.update(self.hyper)
        return hyper


    def build(self, hyper=None, overwrite=False, reload=False):
        """
        Build or rebuild the UVAE model's structure based on the data and constraints provided.

        This method sets up the architecture of the UVAE model, including autoencoders, constraints, and other related
        components. If the model has already been built, this method can be used to update or reload its components based
        on the parameters provided.

        Parameters
        ----------
        hyper : dict, optional
            Dictionary containing custom hyperparameters to update the model's default hyperparameters.
        overwrite : bool, optional
            If True, it will reset the parameters of autoencoders and constraints to new random values, by default False (keep existing parameters).
        reload : bool, optional
            If True, reloads the structure of the model to apply new hyper-parameters, by default False.

        Returns
        -------
        None
        """
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
        """
        Forward and backward propagation through the UVAE model based on given parameters and data.

        This method manages the forward and backward passes for the UVAE model, adjusting weights based on loss and
        training the specified constraints. It calculates the loss for different constraints and updates the model's
        parameters accordingly. Additionally, it can operate in validation mode to evaluate the model's performance
        on validation data.

        Parameters
        ----------
        validation : bool, optional
            If True, performs forward propagation on validation data without backpropagation, by default False.
        batchSize : int, optional
            Number of samples in each batch for training or validation, by default 128.
        sampleLimit : int, optional
            Maximum number of samples to be used during propagation, by default 0 (no limit).
        skipTrained : bool, optional
            If True, skips training of constraints that are already trained, by default True.

        Returns
        -------
        History
            Object containing the history of losses accumulated during training.

        """
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
        """
        Resample the constraints in the UVAE model based on predicted targets.

        Given the current model's predictions, this method adjusts the sampling of the data points
        to achieve a balanced representation. It's particularly useful in cases where the distribution
        of classes in the data is imbalanced, or when it's crucial to have a specific representation
        of samples for particular constraints.

        This internal method is called after each epoch od training.

        Parameters
        ----------
        resample_prop : float, optional
            Proportion of data points to be resampled, adjusted based on the ease_epochs hyper-parameter.
        skipTrained : bool, optional
            If True, skips resampling of constraints that are already trained, by default True.

        Notes
        -----
        The method works by predicting merged targets for each classifier and adjusting the sampling
        of the data points based on these predictions.

        This method is meant to be called internally during training. To add a new resampling source-target pair,
        call source.resample(target) on the individual constraints themselves.
        """
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
        """
        Update the offsets for normalization and standardization constraints in the UVAE model.

        This method recalculates and updates the biases or statistics necessary for normalization
        and standardization constraints in the model. This ensures that the latent representation
        remains centered and scaled appropriately across different batches or modalities.

        Parameters
        ----------
        correction_prop : float, optional
            Proportion of the offsets to be applied, adjusted based on the ease_epochs hyper-parameter.
        skipTrained : bool, optional
            If True, skips updating constraints that are already trained, by default True.

        Notes
        -----
        The method works by iterating over all normalization and standardization constraints in the model.
        For each constraint, it checks if it's time to update (based on the defined intervals) and
        recalculates the necessary biases or statistics. The updated values are then used in the subsequent
        training or inference.
        """
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
        """
        Predict the latent embeddings for the given data map using the model's autoencoders.

        This method returns the latent embeddings for the provided data samples by forwarding them
        through the corresponding autoencoders' encoders.

        Parameters
        ----------
        dataMap : DataMap or dict
            Dictionary mapping data objects to sample indices. The data objects are keys, and
            the associated value is an array of indices specifying which samples to predict embeddings for.
        mean : bool, optional
            If True, use the mean of the latent distribution as the embedding, otherwise
            use a sampled value from the distribution, by default False.
        stacked : bool, optional
            If True, concatenate all embeddings into a single numpy array, by default False.
        bs : int, optional
            Batch size to use when predicting embeddings, by default 4096.

        Returns
        -------
        dict or np.ndarray
            If `stacked` is False, returns a dictionary mapping data objects to their corresponding
            embeddings. If `stacked` is True, returns a single concatenated numpy array of embeddings.

        Notes
        -----
        The method uses the encoder part of the autoencoder to generate the latent embeddings.
        Depending on the `mean` parameter, it either uses the mean of the latent distribution or
        samples from it.
        """
        if not self.built:
            self.build()
        emb = {}
        for data in dataMap:
            emb[data] = self.autoencoders[data].encoder.predictMap({data: dataMap[data]}, mean=mean, bs=bs)[data]
        if stacked:
            emb = np.array(np.concatenate(list(emb.values())))
        return emb


    def reconstruct(self, dataMap, channels:[str]=None, decoderPanels:[Data]=None, conditions:{Labeling:[str]}=None, bs=1024, stacked=False, mean=True):
        """
        Reconstruct data by using the UVAE's decoders.

        Given a set of data points, this method will reconstruct them by predicting through the UVAE's decoders.
        Reconstructed values can either be the original inputs or specific channels merged from different decoders.

        Parameters
        ----------
        dataMap : DataMap or dict
            Dictionary of data objects to sample indices.
        channels : list of str, optional
            List of channels to be reconstructed. If not provided, all channels are used.
        decoderPanels : list of Data, optional
            List of Data objects which decoders should be used for decoding.
            If not provided, all decoders which contain a given channel are used.
        conditions : dict of Normalisation or Labeling to list of str, optional
            Conditions to be set as targets during reconstruction. For each Normalisation or Labeling set as key,
            set a list of target conditions (you can set more than one if not all conditions exist across all data).
        bs : int, optional
            Batch size for processing.
        stacked : bool, optional
            If True, returns a single stacked array of reconstructed values; otherwise, returns a dictionary.
        mean : bool, optional
            If True, uses the mean value of latent samples during reconstruction. Otherwise uses stochastic samples.

        Returns
        -------
        dict or np.array
            Reconstructed data in the format of dictionary or stacked array based on `stacked` parameter.
        """
        if not self.built:
            self.build()
        if channels is not None:
            if len(set(channels)) != len(channels):
                print('Error: channels must be a non-repeating list of channels matching the ones in Data.')
                return None
        rec = {}
        if decoderPanels is None and channels is None: # predict just the original input channels
            for d in dataMap:
                p_dm = {d: dataMap[d]}
                rec[d] = self.autoencoders[d].decoder.batchPrediction(p_dm, mean=mean, bs=bs, conditions=conditions)[d]
        else: # predict specific channels by merging outputs of different decoders
            if channels is None:  # use all channels of specified decoders
                channels = sorted(list(set(np.concatenate([d.channels for d in decoderPanels]))))
            if decoderPanels is None: # use all available decoders
                candidateDecoders = {data: self.autoencoders[data].decoder for data in self.autoencoders}
            else: # use only specified decoders
                candidateDecoders = {data: self.autoencoders[data].decoder for data in decoderPanels}
            useDecoders = {}
            for data in candidateDecoders:
                if len(set(channels).intersection(set(data.channels))):
                    useDecoders[data] = candidateDecoders[data]
            if not len(useDecoders):
                print('Error: no decoders contain specified channels.')
                return None
            else:
                foundChs = set(np.concatenate([d.channels for d in useDecoders]))
                if not all([ch in foundChs for ch in channels]):
                    diff = [ch for ch in channels if ch not in foundChs]
                    print('Error: channels not found in specified decoders: {}'.format(diff))
                    return None
            for d in dataMap:
                p_dm = {d: dataMap[d]}
                accum = {ch: [] for ch in channels}
                for data, dec in useDecoders.items():
                    r = dec.batchPrediction(p_dm, mean=mean, bs=bs, conditions=conditions)[d]
                    for ci, ch in enumerate(data.channels):
                        if ch in accum:
                            accum[ch].append(r[:, ci])
                means = [np.mean(accum[ch], axis=0) for ch in accum]
                rec[d] = np.transpose(np.array(means))
        if stacked:
            return np.array(np.concatenate(list(rec.values())))
        return rec


    def addConstraint(self, const):
        """
        Add a constraint to the UVAE model.

        This method allows for the addition of data or other constraints to the UVAE model.
        If the constraint is of type Data, it will be appended as data. Otherwise, it will be appended as a constraint.

        Parameters
        ----------
        const : Hashable
            Constraint to be added. It can be of type Data or any other constraint type.

        Returns
        -------
        object
            The added constraint with its parent set to the current UVAE instance.
        """
        if type(const) is Data:
            const = self.appendData(const)
        else:
            const = self.appendConstraint(const)
        if const is not None:
            const.parent = self
        return const


    def removeConstraint(self, const):
        """
        Remove a constraint from the UVAE model.

        This method allows for the removal of data or other constraints from the UVAE model.
        All references to the constraint in the UVAE's internal structures will be removed.

        Parameters
        ----------
        const : Hashable
            Constraint to be removed.

        Returns
        -------
        UVAE
            The UVAE instance with the constraint removed.
        """
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
        """
        Create a mapping of all the data in the UVAE model.

        The method returns a DataMap containing indices for each data set in the UVAE model.
        It optionally supports subsampling to retrieve a proportion of the data.

        Parameters
        ----------
        subsample : int, optional
            The number of samples to retrieve. If 0, all samples will be retrieved.
            Otherwise, a proportion of the samples is determined based on the given subsample value.

        Returns
        -------
        DataMap
            A mapping of data sets to indices.
        """
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
        """
        Retrieve constraints of a specific type from the UVAE model.

        Parameters
        ----------
        t : type
            The type of constraints to retrieve.

        Returns
        -------
        list
            A list of constraints of the specified type.
        """
        return [c for c in self.constraints.values() if type(c) is t]


    def allConstraints(self):
        """
        Retrieve all constraints from the UVAE model.

        This method returns a combined list of all non-data and autoencoder constraints in the UVAE model.

        Returns
        -------
        list
            A list of all constraints in the UVAE model.
        """
        return list(self.constraints.values()) + list(self.autoencoders.values())


    def __getitem__(self, name):
        """
        Retrieve a data or constraint by name from the UVAE model.
        """
        for d in self.data:
            if d.name == name:
                return d
        if name in self.constraints:
            return self.constraints[name]
        return None


    def __add__(self, const):
        """
        Use `+` to add a constraint to the UVAE model.
        """
        return self.addConstraint(const)


    def __iadd__(self, const):
        """
        Use `+=` to add a constraint to the UVAE model in-place.
        """
        self + const
        return self


    def __sub__(self, const):
        """
        Use `-` to remove a constraint from the UVAE model.
        """
        return self.removeConstraint(const)


    def __isub__(self, const):
        """
        Use `-=` to remove a constraint from the UVAE model in-place.
        """
        self - const
        return self


    def uniqueName(self, base, names):
        """
        Generate a unique constraint name by appending an index to the given base name.

        Parameters
        ----------
        base : str
            Base string to use for generating the unique name.
        names : list of str
            List of existing names.

        Returns
        -------
        str
            A unique name.
        """
        name = base
        i = 0
        while name in names:
            i += 1
            name = '{}_({})'.format(base, int(i))
        return name


    def appendData(self, const):
        """
        Append a Data instance to the UVAE model.

        Parameters
        ----------
        const : Data
            Data instance to append to the UVAE model.

        Returns
        -------
        Data
            The appended Data instance.
        """
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
        """
        Append a constraint to the UVAE model.

        If the constraint is an Autoencoder with a single mask, it associates the
        autoencoder with the corresponding data. For other constraints, it ensures
        the constraint has a unique name and the appropriate masks and targets set.

        Parameters
        ----------
        const : Constraint
            Constraint instance to append to the UVAE model.

        Returns
        -------
        Constraint
            The appended Constraint instance or None if addition fails.
        """
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
