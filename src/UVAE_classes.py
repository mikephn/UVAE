from src.UVAE_arch import *
import tensorflow.keras.backend as K
import numpy as np
import time
UVAE_DEBUG = 0


class Hashable:
    """
    Base class that provides hashable functionality for constraints.
    It also provides a structure for handling chained adding of constraints to UVAE model.

    Attributes
    ----------
    name : str
        Unique identifier for the constraint, used for hashing and representation.
    _parent : Hashable or None
        Reference to the parent object.

    Methods
    -------
    parent() -> Hashable or None:
        Get the parent of the current object.
    """
    def __init__(self, name):
        self.name = name
        self._parent = None

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __repr__(self):
        return self.name

    def __add__(self, other):
        if self.parent is not None:
            return self.parent + other

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value


class Data(Hashable):
    """
    A class representing a single data stream with related metadata.

    It stores data (X) and optionally its normalized form, along with
    feature names, cached predictions, and validation proportions.

    Attributes
    ----------
    X : numpy array
        The raw column data.
    normed : numpy array or None
        Normalized version of X, if available.
    predictions : dict
        Dictionary caching predictions related to this data.
    channels : list of str
        Names of channels or features in the data.
    valProp : float
        Proportion of data to be used for validation.

    Methods
    -------
    defineValMask():
        Defines a boolean mask for splitting data into validation based on valProp.
    Xn() -> numpy array:
        Returns the normalized data if available, otherwise the raw data.
    """
    def __init__(self, X, name=None, channels=None, valProp=0.0):
        super().__init__(name)
        self.X = X
        self.normed = None
        self.predictions = {}
        if channels is None:
            self.channels = [self.name+'-ch{}'.format(cn) for cn in range(len(X[0]))]
        else:
            self.channels = list(channels)
        self.valProp = valProp

    def defineValMask(self):
        self.valMask = np.zeros(len(self.X), dtype=bool)
        if self.valProp > 0 and len(self.X) > 0:
            n_val = int(round(len(self.X) * self.valProp))
            self.valMask[:n_val] = True
            np.random.shuffle(self.valMask)

    def Xn(self):
        if self.normed is None:
            return self.X
        else:
            return self.normed


class DataMap(dict):
    """
    A dictionary storing references to Data objects and indexes referencing samples in each Data.

    Methods
    -------
    allChannels() -> numpy array:
        Returns list of all unique channels from all Data objects stored in the map.
    """
    def __setitem__(self, key, value):
        if type(key) is Data:
            return super().__setitem__(key, value)
        else:
            print('DataMap is a dictionary of {Data: numpy int array} indexes referencing Data.X.')

    def allChannels(self):
        return np.unique(np.concatenate([d.channels for d in self]))


class Mapping(Hashable):
    """
    A class referencing specific subsets of data.

    This class extends the Hashable class and is designed to keep track of and
    manipulate subsets of the Data class based on masks. These masks determine
    which samples of the Data are considered under this mapping. It provides
    methods to handle, index, and retrieve these subsets efficiently.

    Attributes
    ----------
    masks : dict
        A dictionary mapping Data objects to boolean masks representing the subset of the data.
    _dataMap : dict or None
        A mapping from Data to integer indexes in the data that are under this mapping.
    _reverseMap : dict or None
        A mapping from Data to a dictionary mapping from the index in the data to the index in the mapping.
    _edges : list of int or None
        Accumulative edges for separating different data objects in the flat representation.
    _inds : ndarray or None
        Flat indexes of all data samples under this mapping.
    _valMask : ndarray or None
        A mask indicating which samples are reserved for validation.

    Methods
    -------
    addMask(data: Data, mask: ndarray):
        Add or update a mask for a specific data.
    index():
        Indexes the masks for efficient data retrieval.
    inds(validation: bool = False) -> ndarray:
        Get the indexes of data samples under this mapping.
    length(bs: int, validation: bool = False) -> int:
        Calculate the number of batches given a batch size.
    batch(bs: int, validation: bool = False) -> ndarray:
        Get a random batch of indexes.
    coords(inds: ndarray) -> dict:
        Convert flat indexes to a dictionary of indexes for each data.
    dataMap(inds: ndarray) -> DataMap:
        Convert flat indexes to a DataMap object.
    reverseMap(dataMap: DataMap, undefined=0) -> dict:
        Convert a DataMap object to this mapping's internal representation.
    stack(map: dict) -> ndarray:
        Flatten the data map to a single array.
    Xs(inds: ndarray, normed: bool = True) -> dict:
        Retrieve data samples for the given indexes.
    """
    def __init__(self, name=None, masks=None):
        super().__init__(name)
        self.masks = {}
        if masks is not None:
            if type(masks) is Data:
                masks = [masks]
            if type(masks) is list:
                self.masks.update({d: np.ones(len(d.X), dtype=bool) for d in masks})
            elif type(masks) is dict:
                self.masks.update({d: np.array(masks[d], dtype=bool) for d in masks})
            else:
                print('Error ({}): .masks must either be an array of [Data] or a dictionary mapping {{Data: ndarray}} with binary coverage mask.'.format(self.name))
        self._dataMap = None
        self._reverseMap = None
        self._edges = None
        self._inds = None
        self._valMask = None

    # add reference to a subset of data
    def addMask(self, data, mask):
        self.masks[data] = np.array(mask, dtype=bool)

    # create indexing arrays for added masks
    def index(self):
        if len(self.masks):
            self._dataMap = {}
            self._reverseMap = {}
            valMasks = []
            for i, (data, mask) in enumerate(self.masks.items()):
                p_inds = np.arange(len(mask))[np.array(mask, dtype=bool)]
                self._dataMap[data] = p_inds
                self._reverseMap[data] = {p_inds[n]: n for n in range(len(p_inds))}
                valMasks.append(data.valMask[p_inds])
            self._inds = np.arange(int(np.sum([len(ind) for ind in self._dataMap.values()])))
            self._valMask = np.concatenate(valMasks)
            self._edges = [0]
            for i, ind in enumerate(self._dataMap.values()):
                self._edges.append(self._edges[-1] + len(ind))

    # returns indexes referencing samples under constraint
    def inds(self, validation=False):
        if self._dataMap is None:
            self.index()
        return self._inds[self._valMask == validation]

    def length(self, bs, validation=False):
        return int(np.ceil(len(self.inds(validation=validation))/bs))

    def batch(self, bs, validation=False):
        inds = self.inds(validation=validation)
        shuffle = np.random.randint(0, len(inds), int(bs))
        return inds[shuffle]

    # internal map of indexes masked by the constraint
    def coords(self, inds):
        map = {}
        for i, data in enumerate(self._dataMap):
            m_inds = inds[(inds >= self._edges[i]) & (inds < self._edges[i + 1])] - self._edges[i]
            if len(m_inds):
                map[data] = m_inds
        return map

    # DataMap referencing data samples in global coordinates
    def dataMap(self, inds):
        map = self.coords(inds)
        d_m = DataMap()
        for data in map:
            d_m[data] = self._dataMap[data][map[data]]
        return d_m

    # converts global map to internal, or undefined if outside own mask
    def reverseMap(self, dataMap:DataMap, undefined=0):
        if self._reverseMap is None:
            self.index()
        map = {}
        for data in dataMap:
            if undefined == 0:
                d_inds = np.array(np.repeat(0, len(dataMap[data])), dtype=int)
            else:
                d_inds = np.array(np.repeat(undefined, len(dataMap[data])), dtype=float)
            if data in self.masks:
                included = self.masks[data][dataMap[data]]
                for i, inc in enumerate(included):
                    if inc:
                        d_inds[i] = self._reverseMap[data][dataMap[data][i]]
            map[data] = d_inds
        return map

    # converts map to a flat array
    def stack(self, map):
        if len(map):
            return np.array(np.concatenate(list(map.values())))
        else:
            return np.array([], dtype=float)

    # get X from a subset of self.inds()
    def Xs(self, inds, normed=True):
        d_m = self.dataMap(inds)
        Xs = {}
        for data in d_m:
            if data.normed is None or normed == False:
                Xs[data] = data.X[d_m[data]]
            else:
                Xs[data] = data.normed[d_m[data]]
        return Xs


class Control(Mapping):
    """
    Extends the Mapping class to handle control masks and basic resampling functionality.

    This class is designed to incorporate control masks to the original data masks. The control
    masks can be used to further subset the data or to define specific conditions in which
    data samples are considered. The class also provides functionalities to balance the
    distribution of data samples based on some prediction, achieving an equal representation
    of different classes.

    Attributes
    ----------
    controlMasks : dict
        A dictionary mapping Data objects to boolean masks representing control conditions.
    _ctrlMask : ndarray or None
        Combined control mask obtained from the union of all control masks and the original masks.
    _resampled : ndarray or None
        Indexes of data samples after the resampling operation.

    Methods
    -------
    addControlMask(data: Data, mask: ndarray):
        Add or update a control mask for a specific data.
    calculateControlMask():
        Compute the combined control mask.
    inds(validation: bool = False, controls: bool = True, resampled: bool = True) -> ndarray:
        Get the indexes of data samples under this mapping considering controls and resampling.
    resampledInds(vals_list: list, prop: float = 1.0) -> ndarray:
        Get the indexes of resampled data samples to balance class distribution.
    balance(prediction, prop: float = 1.0):
        Generate resampled data indexes to balance the distribution based on some prediction.
    """
    def __init__(self, controlMasks=None, **kwargs):
        super().__init__(**kwargs)
        self.controlMasks = {}
        if controlMasks is not None:
            self.controlMasks.update(controlMasks)
        self._ctrlMask = None
        self._resampled = None

    def addControlMask(self, data, mask):
        self.controlMasks[data] = mask

    def index(self):
        super(Control, self).index()
        self.calculateControlMask()

    def calculateControlMask(self):
        if len(self.masks):
            ctrlMasks = []
            for i, (data, mask) in enumerate(self.masks.items()):
                d_inds = self._dataMap[data]
                if data in self.controlMasks:
                    combined = mask & self.controlMasks[data]
                    ctrlMasks.append(combined[d_inds])
                else:
                    ctrlMasks.append(np.ones(len(d_inds), dtype=bool))
            self._ctrlMask = np.concatenate(ctrlMasks)

    def inds(self, validation=False, controls=True, resampled=True):
        if self._ctrlMask is None:
            self.index()
        if (not validation) and controls and (self._resampled is not None) and resampled:
            return self._resampled
        else:
            mask = self._valMask == validation
            ctrl_mask = self._ctrlMask == controls
            return self._inds[mask & ctrl_mask]

    # equally sample from each predicted class
    def resampledInds(self, vals_list, prop=1.0):
        n_resamplings = len(vals_list)
        resampled_inds = []
        for r_n in range(n_resamplings):
            vals = vals_list[r_n]
            n_res = len(vals) * prop / n_resamplings
            en = list(set(vals))
            per_class = int(round(n_res / len(en)))
            v_arr = np.array(vals)
            v_inds = np.arange(len(v_arr))
            for e in en:
                e_inds = v_inds[v_arr == e]
                r = np.random.randint(0, len(e_inds), per_class)
                resampled_inds.extend(list(e_inds[r]))
        remaining = int(len(vals_list[0]) - len(resampled_inds))
        if remaining > 0:
            r_inds = np.random.randint(0, len(vals_list[0]), remaining)
            resampled_inds.extend(list(np.arange(len(vals_list[0]))[r_inds]))
        return np.array(resampled_inds, dtype=int)

    def balance(self, prediction, prop=1.0):
        inds = self.inds(validation=False, controls=True, resampled=False)
        if type(prediction) is not list:
            P_cats = [self.stack(prediction)]
        else:
            P_cats = [self.stack(p) for p in prediction]
        res_inds = self.resampledInds(P_cats, prop=prop)
        self._resampled = inds[res_inds]


class Constraint(Control):
    """
    Base class for constraints in the UVAE model.

    Constraints are used to apply specific annotations or conditions on the underlying data subsets
    and can also implement additional trainable network layers that use those annotations. This class
    encapsulates the functionalities needed to define, train, and apply these constraints.

    Attributes
    ----------
    func : callable or None
        The neural network function implementing the constraint.
    loss : callable or None
        Loss function associated with the constraint.
    weight : float
        Weighting factor for the importance of this constraint's loss to model selection.
    frequency : float
        Frequency with which this constraint is trained, by default 1.0.
        This factor scales the number of batches for training relative to other constraints.
        By default, number of batches is proportional to the amount of data each constraint spans.
    trained : bool
        Indicator of whether the constraint is trained or not.
    hyper : dict
        Dictionary storing hyperparameters specific to this constraint.
    saved : dict
        Dictionary to store archived parameters and settings of this constraint.

    Methods
    -------
    archive() -> dict:
        Archive the current state and parameters of this constraint.
    unarchive(d: dict):
        Restore the state and parameters of this constraint from an archive.
    loadParams():
        Load the parameters into the func from the saved state.
    hyperparams() -> dict:
        Retrieve hyperparameters associated with this constraint, including inherited ones.
    outputDim() -> int or None:
        Get the output dimension of the func. Returns None if func is not defined.
    predict(X: ndarray, mean: bool = True) -> ndarray:
        Apply the constraint (func) on input data X and return the transformed data.
    """
    def __init__(self, func=None, loss=None, weight=1.0, frequency=1.0, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.loss = loss
        self.weight = weight
        self.frequency = frequency
        self.trained = False
        self.hyper = {}
        self.saved = {}

    def archive(self):
        self.saved = {'class': type(self).__name__,
             'name': self.name,
             'weight': self.weight,
             'frequency': self.frequency,
             'trained': self.trained,
             'hyper': self.hyper}
        if self.func is not None:
            self.saved['params'] = self.func.get_weights()
        return self.saved

    def unarchive(self, d):
        self.name = d['name']
        self.weight = float(d['weight'])
        self.frequency = float(d['frequency'])
        self.trained = bool(d['trained'])
        self.hyper = d['hyper']
        self.saved = dict(d)

    def loadParams(self):
        if 'params' in self.saved and self.func is not None:
            try:
                self.func.set_weights(self.saved['params'])
            except Exception as e:
                print(self.name, ':', e)

    def hyperparams(self):
        h = dict(self.parent.hyperparams())
        h.update(self.hyper)
        return h

    def outputDim(self):
        if self.func is not None:
            if type(self.func.output) is list:
                return int(self.func.output[-1].shape[-1])
            else:
                return int(self.func.output.shape[-1])
        else:
            return None

    def predict(self, X, mean=True):
        if self.func is not None:
            return self.func(X)
        else:
            return X


class Serial(Constraint):
    """
    Represents a sequence of constraints in the UVAE model, extending the basic Constraint functionalities.

    The `Serial` class is designed to handle the sequential application of multiple constraints,
    for example training a classifier which uses embedding from encoders as its input.
    It also offers functionalities to make batched predictions, and caching of predictions.

    Attributes
    ----------
    embedding : dict or None
        A dictionary storing constraints which should be applied before this constraint for each data stream.
    trainEmbedding : bool
        Indicator of whether the embedding layers should also be trained with this constraint loss.
    adversarial : bool
        If True, and if trainEmbedding is True, the constraint will train the embedding layers in an adversarial manner.
    in_dim : int or None
        Input dimension for the constraint.

    Methods
    -------
    archive() -> dict:
        Archive the current state and parameters of this constraint, including its specific attributes.
    unarchive(d: dict):
        Restore the state and parameters of this constraint from an archive.
    getInput(dataMap: dict, mean: bool = False) -> dict:
        Obtain the representation of data up to this constraint.
    embedMap(dataMap: dict, mean: bool = False, **kwargs) -> tuple:
        Apply the constraint and return both input and prediction.
    batchPrediction(dataMap: dict, mean: bool = True, bs: int = 4096, **kwargs) -> dict:
        Perform batch-wise prediction using the constraint.
    predictMap(dataMap: dict, mean: bool = True, stacked: bool = False, bs: int = 4096, **kwargs) -> dict:
        Return representation after applying the constraint, optionally caching the result.
    cachedPrediction(dataMap: dict) -> dict:
        Retrieve cached predictions for a given data map.
    cachePrediction(dataMap: dict, prediction: dict):
        Store the predictions in cache for future use.
    invalidateCache():
        Invalidate (clear) the cached predictions.
    """
    def __init__(self, embedding=None, trainEmbedding=False, adversarial=False, in_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding if embedding is not None else {}
        self.trainEmbedding = trainEmbedding
        self.adversarial = adversarial
        self.in_dim = in_dim
        if self.adversarial:
            self.trainEmbedding = True

    def archive(self):
        d = super().archive()
        d['trainEmbedding'] = self.trainEmbedding
        d['adversarial'] = self.adversarial
        return d

    def unarchive(self, d):
        super().unarchive(d)
        self.trainEmbedding = d['trainEmbedding']
        self.adversarial = d['adversarial']

    # obtain representation of data up to this constraint
    def getInput(self, dataMap, mean=False):
        emb = {}
        for data in dataMap:
            if data in self.embedding:
                enc = self.embedding[data]
                _, Z = enc.embedMap({data: dataMap[data]}, mean=mean)
                emb[data] = Z[data]
            else:
                if data.normed is None:
                    emb[data] = data.X[dataMap[data]]
                else:
                    emb[data] = data.normed[dataMap[data]]
        return emb

    # apply constraint, return both input and prediction
    def embedMap(self, dataMap, mean=False, **kwargs):
        ins = self.getInput(dataMap, mean)
        outs = {data: self.predict(ins[data], mean=mean) for data in ins}
        return ins, outs

    def batchPrediction(self, dataMap, mean=True, bs=4096, **kwargs):
        predicted = {}
        for data in dataMap:
            n_batches = int(np.ceil(len(dataMap[data]) / bs))
            p_cat = []
            for n_b in range(n_batches):
                b_inds = dataMap[data][int(n_b * bs):int((n_b + 1) * bs)]
                dm = {data: b_inds}
                _, outs = self.embedMap(dm, mean=mean, **kwargs)
                p_cat.append(outs[data])
            if len(p_cat):
                predicted[data] = np.concatenate(p_cat, axis=0)
        return predicted

    # return representation after this constraint is applied, cache if required
    def predictMap(self, dataMap, mean=True, stacked=False, bs=4096, **kwargs):
        if self.parent.shouldCache:
            unavailable = {}
            missing_mask = {}
            cached = self.cachedPrediction(dataMap)
            for d in cached:
                missing = np.any(np.isnan(cached[d]), axis=-1)
                if np.any(missing):
                    unavailable[d] = dataMap[d][missing]
                    missing_mask[d] = missing
            for d in dataMap:
                if d not in cached:
                    unavailable[d] = dataMap[d]
        else:
            unavailable = dataMap

        predicted = self.batchPrediction(unavailable, mean=mean, bs=bs)

        dim = self.outputDim()
        if self.parent.shouldCache and dim is not None:
            self.cachePrediction(unavailable, predicted)
            result = {}
            for d in cached:
                result[d] = cached[d]
                if d in missing_mask:
                    result[d][missing_mask[d]] = predicted[d]
            for d in predicted:
                if d not in result:
                    result[d] = predicted[d]
        else:
            result = predicted
        if stacked:
            return self.stack(result)
        return result

    def cachedPrediction(self, dataMap):
        cached_pred = {}
        for d in dataMap:
            if self.name in d.predictions:
                cached_pred[d] = d.predictions[self.name][dataMap[d]]
                if UVAE_DEBUG > 0:
                    print('Using cached: ', self.name, d.name)
        return cached_pred

    def cachePrediction(self, dataMap, prediction):
        dim = self.outputDim()
        if dim is not None:
            if UVAE_DEBUG > 0:
                print('Caching: ', self.name)
            for d in prediction:
                if not self.name in d.predictions:
                    d.predictions[self.name] = np.zeros((len(d.X), dim), dtype=float) * np.nan
                d.predictions[self.name][dataMap[d]] = prediction[d]

    def invalidateCache(self):
        for d in self.parent.data:
            if self.parent.autoencoders[d].encoder == self:
                # constraint is the encoder for this data type, invalidate all downstream predictions
                if len(d.predictions):
                    d.predictions = {}
                    if UVAE_DEBUG > 0:
                        print('Invalidating cache: ', d.name)
            elif self.name in d.predictions:
                # invalidate only downstream task
                if UVAE_DEBUG > 0:
                    print('Invalidating cache: ', self.name, d.name)
                del d.predictions[self.name]


class Regression(Serial):
    """
    Represents a regression constraint in the UVAE model, usually applied in a serial fashion
    after embedding data to the latent space.

    Attributes
    ----------
    Y : dict
        Dictionary storing the originally supplied (unmasked) targets.
    targets : dict
        Dictionary storing valid regression targets for each data.
    nullLabel : variable type or None
        The label considered as null in the targets supplied to Y.
        The samples with this target are excluded from training.

    Methods
    -------
    setTargets(data: type, Y: type):
        Assign regression targets for the specified data.
    outputDim() -> int:
        Determine the output dimension based on the regression targets.
    build():
        Build the regression layers based on the hyperparameters and architecture defined.
    Ys(inds: list) -> dict:
        Retrieve regression targets for the specified indexes.
    YsFromMap(dataMap: dict, undefined: variable type = None) -> dict:
        Retrieve regression targets based on the data map provided.
    forward(inds: list) -> tuple:
        Execute the forward pass of the regression model for the specified indexes, returning the loss.
    """
    def __init__(self, Y=None, targets=None, nullLabel=None, trainEmbedding=True, **kwargs):
        super().__init__(trainEmbedding=trainEmbedding, **kwargs)
        if Y is None:
            self.Y = {}
        else:
            self.Y = Y
        self.nullLabel = nullLabel
        self.targets = {}
        if targets is not None:
            self.targets = targets

    def setTargets(self, data, Y):
        self.targets[data] = Y

    def outputDim(self):
        out_lens = []
        for d in self.targets:
            ex = self.targets[d][0]
            if type(ex) is np.ndarray:
                out_lens.append(len(ex))
            else:
                out_lens.append(1)
        out_lens = list(set(out_lens))
        if len(out_lens) > 1:
            print('Error: more than one target shape for {}: {}'.format(self.name, out_lens))
        return int(out_lens[0])

    def build(self):
        self.index()
        hyper = self.hyperparams()
        inp = Input((int(self.in_dim),))
        c = MLP(n_dense=hyper['width'],
                relu_slope=hyper['relu_slope'],
                dropout=hyper['dropout'],
                depth=hyper['hidden'],
                out_len=int(self.outputDim()))
        out = c(inp)
        self.func = keras.Model(inp, out)
        self.loss = keras.losses.MSE


    def Ys(self, inds):
        m = self.coords(inds)
        Ys = {data: self.targets[data][m[data]] for data in m}
        return Ys

    def YsFromMap(self, dataMap, undefined=None):
        m = self.reverseMap(dataMap, undefined=np.nan)
        Ys = {}
        for data, inds in m.items():
            Ys[data] = np.array([self.targets[data][int(inds[i])] if not np.isnan(inds[i]) else undefined for i in range(len(inds))])
        return Ys

    def forward(self, inds):
        Map = self.dataMap(inds)
        Zs = self.getInput(Map, mean=True)
        Zcat = tf.concat([Zs[data] for data in Map], axis=0)
        Ys = self.Ys(inds)
        Ycat = tf.concat([Ys[data] for data in Map], axis=0)
        Pcat = self.func(Zcat)
        loss = tf.reduce_mean(self.loss(Ycat, Pcat))
        losses = {self.name: loss}
        weighed_loss = loss * self.weight
        return losses, weighed_loss


class Classification(Regression):
    """
    Extends the 'Regression' constraint to categorical target type.

    This class provides functionalities to enumerate class labels, construct a classification model,
    perform one-hot encoding, and handle class imbalances.

    Attributes
    ----------
    enum : list or None
        List of unique class labels identified from the target data.
    equalizeLabels : bool
        Indicates whether the class distribution should be equalized during training.

    Methods
    -------
    enumerateLabels():
        Identify and store unique class labels from the target data.
    oneHot(Ys: dict) -> dict:
        Convert class labels into one-hot encoded format.
    resample(target: Control):
        Add a target constraint for resampling to handle class imbalances.
    balance(prediction: dict = None, prop: float = 1.0):
        Calculate indexes to balance the class distribution in the training data.
    categorize(emb: dict, softmax: bool = True, called: bool = True, stacked: bool = False) -> dict:
        Convert softmax outputs or logits to class labels.
    predictMap(dataMap: dict, softmax: bool = True, mean: bool = True, called: bool = True, stacked: bool = False, bs: int = 4096) -> dict:
        Predict class labels for the given data map.
    """
    def __init__(self, Y=None, equalizeLabels=False, **kwargs):
        super().__init__(Y=Y, **kwargs)
        self.enum = None
        self.equalizeLabels = equalizeLabels

    def index(self):
        super(Classification, self).index()
        self.outputDim()

    def enumerateLabels(self):
        labs = np.concatenate(list(self.targets.values()))
        self.enum = sorted(list(set(labs)))

    def outputDim(self):
        if self.enum is None and len(self.targets):
            self.enumerateLabels()
        return int(len(self.enum))

    def build(self):
        self.index()
        super(Classification, self).build()
        self.loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    def oneHot(self, Ys):
        Yoh = {}
        for data in Ys:
            Y = Ys[data]
            oh = np.zeros((len(Y), len(self.enum)), dtype=float)
            for i, c in enumerate(self.enum):
                oh[Y == c, i] = 1
            Yoh[data] = oh
        return Yoh

    def Ys(self, inds, called=False):
        Ys = super(Classification, self).Ys(inds)
        if called:
            return Ys
        return self.oneHot(Ys)

    def resample(self, target:Control):
        if self.parent is None:
            print('Add {} to UVAE before setting resampling target.')
        else:
            if not self in self.parent.resamplings:
                self.parent.resamplings[self] = set()
            self.parent.resamplings[self].add(target)

    def balance(self, prediction=None, prop=1.0):
        if prediction is not None:
            super(Classification, self).balance(prediction, prop=prop)
        else:
            # balance labels
            inds = self.inds(validation=False, controls=True, resampled=False)
            Ys = self.Ys(inds, called=True)
            super(Classification, self).balance([Ys], prop=prop)

    def archive(self):
        d = super().archive()
        d['enum'] = self.enum
        return d

    def unarchive(self, d):
        super().unarchive(d)
        self.enum = d['enum']

    def categorize(self, emb, softmax=True, called=True, stacked=False):
        if softmax:
            def smax(X):
                exp_max = np.exp(X - np.max(X, axis=-1, keepdims=True))
                return exp_max / np.sum(exp_max, axis=-1, keepdims=True)
            for d in emb:
                emb[d] = smax(emb[d])
        if called:
            emb = {data: np.argmax(emb[data], axis=-1) for data in emb}
            if self.enum is not None:
                emb = {data: np.array([self.enum[int(i)] for i in emb[data]]) for data in emb}
        if stacked:
            return self.stack(emb)
        return emb

    def predictMap(self, dataMap, softmax=True, mean=True, called=True, stacked=False, bs=4096):
        res = super().predictMap(dataMap, mean=mean, bs=bs, stacked=False)
        return self.categorize(res, softmax=softmax, called=called, stacked=stacked)


class Labeling(Classification):
    """
    The 'Labeling' class is used for fixed non-trainable categorical assignments,
    such as known labellings, batch assignments etc. Unlike 'Classification' it does not
    implement a prediction network, but instead always returns the provided labels.
    """
    def __init__(self, nullLabel=None, **kwargs):
        super().__init__(nullLabel=nullLabel, **kwargs)
        self.trained = True

    def predictMap(self, dataMap, called=True, stacked=False, **kwargs):
        Ys = self.YsFromMap(dataMap, undefined=self.nullLabel)
        if called == False:
            Ys = self.oneHot(Ys)
        if stacked:
            Ys = self.stack(Ys)
        return Ys

    def forward(self, inds):
        return {}, 0


class Encoder(Serial):
    """
    The Encoder class is used to encode input data into latent representations.
    Conditional encoding is supported by concatenating the condition representations to the inputs.
    Can also be used to subset the data input to desired columns only.

    Attributes
    ----------
    channels : list or None
        List of feature names corresponding to input columns.
    channelMaps : dict or None
        Dictionary mapping data columns to supported input channels.
    condFuncs : dict
        Conditional functions to be appended to the input data before encoding.

    Methods
    -------
    build(in_dim: int, n_dense: int, relu_slope: float, dropout: float, depth: int, out_len: int, variational: bool = False, categorical: bool = False, condFuncs: dict = None) -> tuple:
        Build the encoder model based on the provided specifications.
    getInput(dataMap: dict, mean: bool = False) -> dict:
        Obtain the input representation for the encoder.
    embedMap(dataMap: dict, mean: bool = False, **kwargs) -> tuple:
        Apply the encoder to the provided data map and return both the input and encoded representations.

    """
    def __init__(self, channels=None, channelMaps=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.channelMaps = channelMaps
        self.condFuncs = {}


    def build(self, in_dim, n_dense, relu_slope, dropout, depth, out_len, variational=False, categorical=False, condFuncs=None):
        """
        Build the encoder based on the provided specifications.

        Parameters
        ----------
        in_dim : int
            Input dimensionality.
        n_dense : int
            Number of dense units.
        relu_slope : float
            Slope for the LeakyReLU activation function.
        dropout : float
            Dropout rate.
        depth : int
            Depth of the MLP.
        out_len : int
            Output length.
        variational : bool, optional
            If True, the encoder is built as a variational encoder.
        categorical : bool, optional
            If True, the encoder output undergoes a softmax activation.
        condFuncs : dict, optional
            Conditional functions to be applied to the input data.

        Returns
        -------
        tuple
            A tuple containing the input tensor and the encoded output tensor.
        """
        inp = Input((int(in_dim),))
        self.condFuncs = condFuncs
        c_ins = []
        if len(self.condFuncs) == 0:
            in_cat = inp
        else:
            ins = [inp]
            for c_func in self.condFuncs.values():
                c_ins.append(c_func.input)
                ins.append(c_func.output)
            in_cat = tf.concat(ins, axis=-1)
        enc = MLP(n_dense=n_dense,
                  relu_slope=relu_slope,
                  dropout=dropout,
                  depth=depth,
                  out_len=out_len)
        if variational:
            enc = GaussianEncoder(encoder=enc,
                                  latent_len=out_len,
                                  input_len=out_len)
            z_mean, z_log_var, z = enc(in_cat)
            self.func = keras.Model([inp] + c_ins, [z_mean, z_log_var, z], name=self.name.replace(" ", "_"))
        else:
            z = enc(in_cat)
            if categorical:
                z = keras.layers.Activation('softmax')(z)
            self.func = keras.Model([inp] + c_ins, z, name=self.name.replace(" ", "_"))
        return inp, z


    def getInput(self, dataMap, mean=False):
        """
        Obtain the input representation for the encoder, considering only selected channels.

        Parameters
        ----------
        dataMap : dict
            Dictionary mapping data types to their respective representations.
        mean : bool, optional
            If True, the mean representation is obtained.

        Returns
        -------
        dict
            Dictionary containing the input representations.
        """
        Xs = super(Encoder, self).getInput(dataMap, mean=mean)
        for data in Xs:
            if (self.channelMaps is not None) and (data in self.channelMaps):
                Xs[data] = tf.gather(Xs[data], self.channelMaps[data], axis=-1)
        return Xs


    def embedMap(self, dataMap, mean=False, **kwargs):
        """
        Apply the encoder to the provided data map. If conditional encoding is used,
        the condition representations are appended to the input before encoding.

        Parameters
        ----------
        dataMap : dict
            Dictionary mapping data types to their respective representations.
        mean : bool, optional
            If True, the mean representation is obtained.

        Returns
        -------
        tuple
            A tuple containing dictionaries of the input and encoded representations.
        """
        ins = self.getInput(dataMap, mean)
        outs = {}
        for data in dataMap:
            enc_inp = [ins[data]]
            if len(self.condFuncs):
                cs = [c.predictMap({data: dataMap[data]}, stacked=True, mean=True, called=False)
                      for c in self.condFuncs]
                enc_inp.extend(cs)
            outs[data] = self.predict(enc_inp, mean=mean)
        return ins, outs


class Unbiasing(Encoder):
    """
    The Unbiasing class extends the functionalities of the `Encoder` class to adjust the encoded representation
    by subtracting the biases of conditions specified using the 'Normalisation' constraints.

    Attributes
    ----------
    offsets : dict
        Dictionary that maps conditions to mean offsets.

    Methods
    -------
    embedMap(dataMap: dict, mean: bool = False) -> tuple:
        Apply the embedding and unbiasing transformation to the provided data map and return both the input and corrected representations.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.offsets = {}

    def build(self, **kwargs):
        self.offsets = {}
        return super(Unbiasing, self).build(**kwargs)

    def predict(self, X, mean=False):
        """
        Predict the latent embedding and returns either mean or sampled representation.

        Parameters
        ----------
        X : array_like
            Input data.
        mean : bool, optional
            If True, returns the mean representation.

        Returns
        -------
        array_like
            The encoded representation of the data.

        Notes
        -----
        This function does not apply the latent space normalisation. To obtain latent embedding in the
        unbiased form (with batch offsets subtracted) use the 'predictMap' function, which internally calls 'embedMap'.
        """
        m, v, z = self.func(X)
        if mean:
            return m
        else:
            return z


    def embedMap(self, dataMap, mean=False):
        """
        Embed the data to latent space and apply the unbiasing transformations.

        Parameters
        ----------
        dataMap : dict
            Subset of data objects and indexes to embed.
        mean : bool, optional
            If True, the mean representation is obtained.

        Returns
        -------
        tuple
            A tuple containing dictionaries of the input and the corrected latent representations.
        """
        Xs, Zs = super(Unbiasing, self).embedMap(dataMap, mean=mean)
        corr = {data: np.zeros(tuple(Zs[data].shape), dtype=float) for data in Zs}
        for const in self.offsets:
            if type(const) is Normalization:
                batch_ids = const.YsFromMap(dataMap, undefined=None)
                for data, b_ids in batch_ids.items():
                    for i, b_id in enumerate(b_ids):
                        if b_id is not None:
                            if b_id in self.offsets[const]:
                                bias = self.offsets[const][b_id]
                                corr[data][i] -= bias
        Zcorr = {data: Zs[data] + corr[data] for data in Zs}
        return Xs, Zcorr

    def archive(self):
        d = super(Unbiasing, self).archive()
        d['offsets'] = {str(const): self.offsets[const] for const in self.offsets}
        return d

    def unarchive(self, d):
        super().unarchive(d)
        self.offsets = d['offsets']


class Decoder(Serial):
    """
    The Decoder class with optional conditioning appended to the latent space input.

    Attributes
    ----------
    condFuncs : dict
        Dictionary that maps conditions to functions used for conditioning.
    autoencoder : Autoencoder object
        Associated autoencoder that contains this decoder.

    Methods
    -------
    build(in_dim, n_dense, relu_slope, dropout, depth, out_len, condFuncs=None) -> tuple:
        Build the decoder based on the provided specifications.
    embedMap(dataMap: dict, mean: bool = False, conditions: dict = None) -> tuple:
        Decode the provided data map, conditioned on certain labels.

    Notes
    -----
    The Decoder class is designed to decode encoded representations back to their original space.
    It can be conditioned on certain labels to produce specific outputs.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.condFuncs = {}
        self.autoencoder = None

    def build(self, in_dim, n_dense, relu_slope, dropout, depth, out_len, condFuncs=None):
        """
        Build the decoder based on the provided specifications.

        Parameters
        ----------
        in_dim : int
            Dimension of the input.
        n_dense : int
            Number of dense units in the MLP.
        relu_slope : float
            Slope of the ReLU activation function.
        dropout : float
            Dropout rate.
        depth : int
            Depth of the MLP.
        out_len : int
            Length of the output.
        condFuncs : dict, optional
            Dictionary of conditioning functions.

        Returns
        -------
        tuple
            A tuple containing the input tensor, conditioning input tensors, and the decoded output tensor.
        """
        self.condFuncs = condFuncs
        z_inp = Input((int(in_dim),))
        c_ins = []
        if len(self.condFuncs) == 0:
            z_cat = z_inp
        else:
            zs = [z_inp]
            for c_func in self.condFuncs.values():
                c_ins.append(c_func.input)
                zs.append(c_func.output)
            z_cat = tf.concat(zs, axis=-1)
        dec = MLP(n_dense=n_dense,
                  relu_slope=relu_slope,
                  dropout=dropout,
                  depth=depth,
                  out_len=int(out_len))
        out = dec(z_cat)
        self.func = keras.Model([z_inp] + c_ins, out, name=self.name.replace(" ", "_"))
        self.loss = keras.losses.MSE
        return z_inp, c_ins, out

    def embedMap(self, dataMap, mean=False, conditions:{Labeling:[str]}=None):
        """
        Decode the provided data map, optionally appending conditioning.

        Parameters
        ----------
        dataMap : dict
            Subset of data objects and indexes to embed.
        mean : bool, optional
            If True, the mean representation is used for decoding.
        conditions : dict, optional
            Dictionary mapping conditions to list of targets.

        Returns
        -------
        tuple
            A tuple containing dictionaries of the input and the decoded representations.
        """
        # In case of straight through reconstruction, without changing batch target:
        if np.all([d in self.autoencoder.masks for d in dataMap]) and conditions is None:
            _, Rec = self.autoencoder.embedMap(dataMap, mean=mean)
            return None, Rec

        # In case of cross-panel prediction, or manually setting batch target:
        # When using normalisation or conditioning you should always specify a target batch,
        # because the decoders aren't ever trained on unbiased encodings.
        # This function selects a target batch by comparing specified list with available source batches:
        def selectBatch(const, available):
            if conditions is None or const not in conditions:
                if available is None or len(available) == 0:
                    # No target batches.
                    return None
                else:
                    # Not specified but batches are available, use the first one.
                    return available[0]
            specified = conditions[const]
            if type(specified) is str:
                specified = [specified]
            matching = [b for b in specified if b in available]
            if len(matching) == 0:
                # No match between specified and available batches.
                return None
            else:
                # One or more matches, pick first batch from matches.
                return matching[0]

        ins = self.getInput(dataMap, mean) # get unbiased latent encoding of samples from source encoders
        # If normalisation is present add target batch offsets:
        if hasattr(self.autoencoder.encoder, 'offsets'):
            bias = np.zeros(list(ins.values())[0].shape[1], dtype=float) # sum bias from multiple conditions
            for norm in self.autoencoder.encoder.offsets:
                useBatch = selectBatch(norm, list(self.autoencoder.encoder.offsets[norm].keys()))
                #print(self.name, norm.name, useBatch)
                if useBatch is not None:
                    bias += self.autoencoder.encoder.offsets[norm][useBatch]
            for data in ins:
                ins[data] = ins[data] + np.tile(bias, (len(ins[data]), 1))

        # If conditional decoders are used, set conditioning inputs:
        outs = {}
        for data in dataMap:
            dec_inp = [ins[data]]
            if len(self.condFuncs):
                cs = []
                for c in self.condFuncs:
                    useBatch = selectBatch(c, c.enum)
                    oh = np.zeros(len(c.enum), dtype=int)
                    if useBatch is not None:
                        # using the specified decoding condition, else zeros will be used
                        oh[int(list(c.enum).index(useBatch))] = 1
                    cs.append(np.tile(oh, (len(dataMap[data]), 1)))
                dec_inp.extend(cs)
            outs[data] = self.func(dec_inp)
        return ins, outs


class Autoencoder(Serial):
    """
    The Autoencoder class that combines encoding and decoding functionalities.

    This class offers the ability to build both standard and variational autoencoders,
    with optional conditioning of encoders and decoders on specific labels. The encoder can optionally
    be 'Unbiasing', which applies the latent space normalisation before returning latent representations.

    Methods
    -------
    build():
        Constructs the autoencoder by building both the encoder and the decoder parts.
    """
    def __init__(self, name, conditions:[Classification]=None, condEncoder=True, in_dim=None, latent_dim=None,
                 variational=True, categorical=False, **kwargs):
        """
                Initializes the Autoencoder with the given parameters.

                Parameters
                ----------
                name : str
                    The name of the autoencoder.
                conditions : list, optional
                    List of conditions to which the autoencoder will be conditioned upon.
                condEncoder : bool, default=True
                    If True, the encoder is conditioned together with decoder.
                in_dim : int
                    Dimension of the input data.
                latent_dim : int, optional
                    Dimension of the latent space.
                variational : bool, default=True
                    If True, constructs a variational autoencoder.
                categorical : bool, default=False
                    If True, the latent representation is categorical.
        """
        super().__init__(name=name, in_dim=in_dim, **kwargs)
        self.latent_dim = latent_dim
        self.variational = variational
        self.categorical = categorical
        self.conditions = conditions
        self.condFuncs = {}
        self.condEncoder = condEncoder
        if self.variational:
            self.encoder = Unbiasing(name=self.name + '-encoder')
        else:
            self.encoder = Encoder(name=self.name + '-encoder')
        self.decoder = Decoder(name=self.name + '-decoder')
        self.decoder.autoencoder = self

    def build(self):
        self.index()
        hyper = self.hyperparams()
        if self.latent_dim is None:
            self.latent_dim = hyper['latent_dim']
        self.beta = hyper['beta']
        if self.conditions is not None:
            for c in self.conditions:
                c_len = c.outputDim()
                c_out = c_in = Input((c_len,))
                if hyper['cond_dim'] > 0:
                    emb = MLP(n_dense=hyper['cond_width'],
                              depth=hyper['cond_hidden'],
                              out_len=hyper['cond_dim'],
                              dropout=0)
                    c_out = emb(c_in)
                self.condFuncs[c] = keras.Model(c_in, c_out)

        inp, z = self.encoder.build(in_dim=self.in_dim,
                                    n_dense=hyper['width'],
                                    relu_slope=hyper['relu_slope'],
                                    dropout=hyper['dropout'],
                                    depth=hyper['hidden'],
                                    out_len=self.latent_dim,
                                    variational=self.variational,
                                    categorical=self.categorical,
                                    condFuncs=self.condFuncs if self.condEncoder else {})

        z_inp, c_ins, out = self.decoder.build(in_dim=self.latent_dim,
                                 n_dense=hyper['width'],
                                 relu_slope=hyper['relu_slope'],
                                 dropout=hyper['dropout'],
                                 depth=hyper['hidden'],
                                 out_len=int(self.in_dim),
                                 condFuncs=self.condFuncs)

        self.func = keras.Model([inp] + c_ins, self.decoder.func([z]+c_ins))
        self.decoder.embedding = {dt: self.encoder for dt in self.masks}

    def archive(self):
        d = super(Autoencoder, self).archive()
        d['input'] = {data.name: data.channels for data in self.masks}
        if type(self.encoder.func.input) is list:
            d['in_dim'] = int(self.encoder.func.input[0].shape[-1])
        else:
            d['in_dim'] = int(self.encoder.func.input.shape[-1])
        d['latent_dim'] = self.latent_dim
        d['variational'] = self.variational
        d['categorical'] = self.categorical
        d['condEncoder'] = self.condEncoder
        if self.conditions is not None:
            d['conditions'] = [cond.name for cond in self.conditions]
        d_enc = self.encoder.archive()
        if 'offsets' in d_enc:
            d['offsets'] = d_enc['offsets']
        return d

    def unarchive(self, d):
        super().unarchive(d)
        self.in_dim = d['in_dim']
        self.latent_dim = d['latent_dim']
        self.encoder.trained = self.trained
        self.categorical = d['categorical']
        self.condEncoder = d['condEncoder']

    def loadParams(self):
        super().loadParams()
        if type(self.encoder) is Unbiasing and 'offsets' in self.saved:
            self.encoder.offsets = {}
            for bias in self.parent.constraintsType(Normalization):
                if bias.name in self.saved['offsets']:
                    self.encoder.offsets[bias] = self.saved['offsets'][bias.name]

    @Hashable.parent.setter
    def parent(self, value):
        self._parent = value
        self.encoder.parent = value
        self.decoder.parent = value

    def embedMap(self, dataMap, mean=False, **kwargs):
        ins = self.encoder.getInput(dataMap, mean=mean)
        outs = {}
        for data in dataMap:
            conds = []
            if len(self.condFuncs):
                cs = [c.predictMap({data: dataMap[data]}, stacked=True, mean=True, called=False)
                      for c in self.condFuncs]
                conds.extend(cs)
            enc_inp = [ins[data]]
            if self.condEncoder:
                enc_inp += conds
            Z = self.encoder.predict(enc_inp, mean=mean)
            dec_inp = [Z] + conds
            rec = self.decoder.predict(dec_inp, mean=mean)
            outs[data] = rec
        return ins, outs

    def forward(self, inds, mean=False):
        Map = self.dataMap(inds)
        Xs, Recs = self.embedMap(Map, mean=mean)
        X_cat = tf.concat([Xs[d] for d in Map], axis=0)
        Rec_cat = tf.concat([Recs[d] for d in Map], axis=0)
        losses = {}
        weighed_loss = 0
        rec_loss = tf.reduce_mean(self.decoder.loss(X_cat, Rec_cat))
        weighed_loss += rec_loss * self.decoder.weight * self.weight
        losses[self.name] = rec_loss
        # regularization losses can only be accessed after forward pass
        if len(self.encoder.func.losses):
            kl_loss = losses[self.name + '-kl'] = self.encoder.func.losses[0] * self.beta
            weighed_loss += kl_loss * self.encoder.weight * self.weight
            losses[self.name + '-rec'] = rec_loss
            losses[self.name] += kl_loss
        return losses, weighed_loss


class Subspace(Autoencoder):
    """
    The Subspace class extends the Autoencoder class to embed data into shared subspaces.

    This class is designed for scenarios where different datasets share common channels or features.
    It provides functionality to map these shared channels and ensure that their embeddings are aligned
    across datasets.

    Attributes
    ----------
    channels : list
        List of shared channels across datasets.
    channelMaps : dict
        Dictionary mapping each dataset channels to subset of shared channels.
    pull : float
        Weight for the merging loss that pulls embeddings of shared channel latent representations together.

    Methods
    -------
    build():
        Constructs the subspace autoencoder and sets up the shared channels.
    mapSharedChannels():
        Identifies and maps the shared channels across datasets.
    forward(inds, mean=False) -> tuple:
        Computes the forward pass of the subspace autoencoder, including the merging loss.
    """
    def __init__(self, channels=None, variational=True, trainEmbedding=True, pull=1.0, **kwargs):
        super().__init__(in_dim=None, variational=variational, trainEmbedding=trainEmbedding, **kwargs)
        self.channels = channels
        self.channelMaps = {}
        self.pull = pull

    def build(self):
        super(Subspace, self).index()
        self.mapSharedChannels()
        self.in_dim = len(self.channels)

        super().build()
        self.encoder.channels = self.channels
        self.encoder.channelMaps = self.channelMaps
        self.loss = keras.losses.MSE

    def archive(self):
        d = super().archive()
        d['pull'] = self.pull
        return d

    def unarchive(self, d):
        super().unarchive(d)
        self.pull = d['pull']

    def mapSharedChannels(self):
        """
        Constructs the subspace autoencoder and sets up the shared channels.

        This method first identifies the shared channels across datasets and then builds the autoencoder
        to embed data into a shared subspace.
        """
        d_chs = [p.channels for p in self.masks.keys()]
        if self.channels is None:
            shared = set(d_chs[0])
            for ch in d_chs[1:]:
                shared = shared.intersection(set(ch))
            self.channels = sorted(list(shared))
            print('{}: {}'.format(self.name, self.channels))
        if len(self.channels) == 0 or len(self.masks) < 2:
            print('No shared channels found. Specify at least two Data inputs with shared channels.')
            for d in d_chs:
                print(d)
        for data in self.masks:
            col_inds = np.array([data.channels.index(c) for c in self.channels], dtype=int)
            self.channelMaps[data] = col_inds

    def forward(self, inds, mean=False):
        losses, weighed_loss = super().forward(inds, mean=mean)
        if len(self.embedding) and self.trainEmbedding:
            Map = self.dataMap(inds)
            _, Z_self = self.encoder.embedMap(Map, mean=True)
            Z_cat_self = tf.concat([Z_self[data] for data in Map], axis=0)
            Zs_other = self.getInput(Map, mean=True)
            Z_cat_other = tf.concat([Zs_other[data] for data in Map], axis=0)
            merge_loss = tf.reduce_mean(self.loss(Z_cat_self, Z_cat_other))
            losses[self.name + '-merge'] = merge_loss * self.pull
            weighed_loss += merge_loss * self.weight
        return losses, weighed_loss


class Projection(Autoencoder):
    """
    The Projection class implements an Autoencoder to project data from latent space to a lower dimensional
    space (2D by default), typically for visualisation purposes.

    Notes
    -----
    When adding 'Projection' as a constraint to a UVAE model, an autoencoder is created which takes latent representations
    as input, and projects them to 2D space. By default, this does not affect the original embeddings (trainEmbedding is set to False).
    """
    def __init__(self, latent_dim=2, trainEmbedding=False, variational=False, **kwargs):
        """
        Initializes the Projection with the given parameters.

        Parameters
        ----------
        latent_dim : int, default=2
            Dimension of the latent space. Typically set to 2 or 3 for visualization purposes.
        trainEmbedding : bool, default=False
            If True, the Projection loss is backpropagated through the main encoders in training.
        variational : bool, default=False
            If True, constructs a variational autoencoder.
        """
        super().__init__(latent_dim=latent_dim,
                         variational=variational,
                         trainEmbedding=trainEmbedding, **kwargs)


class Normalization(Labeling):
    """
    The Normalization class extends the Labeling class to handle normalization operations, specifically in the
    context of batch effect correction.

    This class is designed to calculate and correct for batch-specific biases in the data, working with 'Unbiasing'
    class of encoders, thereby ensuring that data across different batches are comparable.

    Attributes
    ----------
    target : str, optional
        The name of the target batch to which other batches will be aligned.
    balanceBatchAvg : bool, default=True
        If True, balance control data to equalize proportions between batches. If False, use equal class proportions.
    interval : int, default=1
        The interval for normalization operations.
    useClasses : list, optional
        A list of classes to include when resampling the constraint.
        Classes are a subset of the resampling source predictions (e.g. cell types, not a subset of batches).
    trained : bool, default=False
        Indicates if the constraint has been trained (and offsets should no longer be updated).

    Methods
    -------
    resampledInds(vals_list: list, prop: float=1.0, dropMissing: bool=False) -> np.ndarray:
        Resample indices based on given class labeling(s).
    calculateBias(encoders: dict, prop: float=1.0):
        Update batch-specific biases in the latent space.

    """
    def __init__(self, target=None, balanceBatchAvg=True, interval=1, useClasses:list=None, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.interval = self.intervalsLeft = int(interval)
        self.balanceBatchAvg = balanceBatchAvg
        self.useClasses = useClasses
        self.trained = False

    def hyperparams(self):
        h = super(Normalization, self).hyperparams()
        del h['hidden']
        del h['width']
        del h['frequency']
        return h

    def archive(self):
        d = super().archive()
        d['interval'] = self.interval
        d['balanceBatchAvg'] = self.balanceBatchAvg
        return d

    def unarchive(self, d):
        super().unarchive(d)
        self.interval = d['interval']
        self.balanceBatchAvg = d['balanceBatchAvg']

    def resampledInds(self, vals_list, prop=1.0, dropMissing=False):
        """
        Resamples indices based on given class labelling(s). This function is used to
        ensure that the samples from different classes are represented equally (or in
        the same proportion across batches) when calculating the batch normalisation offsets.

        Parameters
        ----------
        vals_list : list
            A list containing class predictions for the normalised data to be used for resampling.
            If more than one assignment is given, an equal portion of the data is resampled using each assignment.
        prop : float, default=1.0
            Proportion of data resampled (increased each epoch depending on the 'ease_epochs' hyper-parameter).
        dropMissing : bool, default=False
            If True, classes that are missing in any batch will be ignored during resampling.

        Returns
        -------
        np.ndarray
            An array of indices corresponding to the resampled data.
        """
        inds = self.inds(validation=False, controls=True, resampled=False)
        B = self.stack(self.Ys(inds, called=True))
        all_inds = np.arange(len(B))

        n_resamplings = len(vals_list)
        val_props = []
        for r_n in range(n_resamplings):
            vals = vals_list[r_n]
            v_arr = np.array(vals)
            classes = list(set(v_arr))
            if self.useClasses is not None:
                classes = [c for c in classes if c in self.useClasses]
            if not self.balanceBatchAvg:
                # sample equal number per class (up-sample rare classes, down-sample frequent)
                avg_prop = {c: 1.0/len(classes) for c in classes}
            else:
                # determine proportion that's average to all batches and apply to every batch
                missing = {c: [] for c in classes}
                counts = {c: [] for c in classes}
                for b_id in self.enum:
                    b_inds = all_inds[B == b_id]
                    b_vals = v_arr[b_inds]
                    for c in classes:
                        b_ct = np.sum(b_vals == c)
                        if b_ct == 0:
                            missing[c].append(b_id)
                        else:
                            counts[c].append(b_ct)
                if dropMissing:
                    for c, missing_batches in missing.items():
                        if len(missing_batches):
                            if UVAE_DEBUG:
                                print('{} not predicted while resampling batches: {}'.format(c, missing_batches))
                            if c in counts:
                                del counts[c]
                total = np.sum([np.sum(counts[c]) for c in counts])
                avg_prop = {c: np.sum(counts[c])/total for c in counts}
            val_props.append(avg_prop)

        # resample (prop) of each batch to the average class proportions across all batches
        resampled_inds = []
        for r_n in range(n_resamplings):
            vals = vals_list[r_n]
            v_arr = np.array(vals)
            avg_prop = val_props[r_n]
            for b_id in self.enum:
                b_inds = all_inds[B == b_id]
                b_vals = v_arr[b_inds]
                b_inds_res = []
                # maintain the same total number of samples in each batch
                b_n_samples = len(b_inds)
                # find which classes are present in the resampled group
                set_b_vals = set(b_vals)
                present_props = {}
                for c in avg_prop:
                    if c in set_b_vals:
                        present_props[c] = avg_prop[c]
                sum_props_present = np.sum(list(present_props.values()))
                if sum_props_present > 0:
                    for c in present_props:
                        present_props[c] = present_props[c] / sum_props_present
                # sample defined proportions
                for c, c_prop in present_props.items():
                    bc_inds = b_inds[b_vals == c]
                    if len(bc_inds):
                        to_sample = int(np.round(b_n_samples * c_prop * prop / n_resamplings))
                        r = np.random.randint(0, len(bc_inds), to_sample)
                        b_inds_res.extend(list(bc_inds[r]))
                # fill the remaining (1-prop) with random samples
                remaining = int((float(b_n_samples) / n_resamplings) - len(b_inds_res))
                if remaining > 0:
                    r_inds = np.random.randint(0, len(b_inds), remaining)
                    b_inds_res.extend(list(b_inds[r_inds]))
                resampled_inds.extend(b_inds_res)
        return np.array(resampled_inds, dtype=int)


    def calculateBias(self, encoders: {Data: Unbiasing}, prop=1.0):
        """
        Calculates batch-specific biases in the latent embedding using the provided encoders. These biases
        are then used after encoding to adjust the data such that it is more consistent across batches.
        The function computes the bias for each batch relative to a target batch or the average
        of all batches, and saves the offsets independently for each encoder.

        Parameters
        ----------
        encoders : dict
            A dictionary mapping data to their corresponding Unbiasing encoders.
        prop : float, default=1.0
            Proportion by which the calculated bias should be applied.
            For instance, prop=0.5 would apply half of the calculated bias.

        Notes
        -----
        The function modifies the 'offsets' attribute of the provided encoders in-place,
        adding or updating the calculated biases.
        """
        inds = self.inds(controls=True, resampled=True)
        Map = self.dataMap(inds)
        Bs = self.Ys(inds)
        batchZs = {b_id: [] for b_id in self.enum}
        for data in Map:
            if data in encoders:
                enc = encoders[data]
                if self in enc.offsets:
                    # Erase the existing offset corrections before calculating new offsets
                    del enc.offsets[self]
                Zs = enc.predictMap({data: Map[data]}, mean=True)
                Z = Zs[data]
                for b_i, b_name in enumerate(batchZs):
                    bZ = Z[np.array(Bs[data][:, b_i], dtype=bool)]
                    if len(bZ):
                        batchZs[b_name].append(bZ)
        means = {b_id: np.mean(np.concatenate(batchZs[b_id]), axis=0) for b_id in batchZs if len(batchZs[b_id])}
        if self.target is not None:
            if self.target not in means:
                print('No samples for target: {}'.format(self.target))
                return
            target = means[self.target]
        else:
            target = np.mean(list(means.values()), axis=0)
        offsets = {b_id: (means[b_id] - target)*prop for b_id in means}
        for d, enc in encoders.items():
            enc_batches = {b: offsets[b] for b in offsets if (d in Map and b in self.Y[d])}
            enc.offsets[self] = enc_batches


class Standardization(Normalization):
    def __init__(self, **kwargs):
        """
        Initializes the Standardization class, which is responsible for performing
        batch-wise standardization (z-score normalization) on the data.
        """
        super().__init__(**kwargs)
        self.stats = {}

    def archive(self):
        d = super().archive()
        d['stats'] = self.stats
        return d

    def unarchive(self, d):
        super().unarchive(d)
        self.stats = d['stats']

    def loadParams(self):
        super().loadParams()
        self.standardizeData()

    def calculateStats(self):
        """
        Calculates the mean and standard deviation for each batch in the data.
        The results are stored in the 'stats' attribute, which maps batch IDs
        to their corresponding statistics.
        """
        inds = self.inds(controls=True, resampled=True)
        Map = self.dataMap(inds)
        Xs = self.Xs(inds, normed=False)
        Bs = self.Ys(inds)
        batchXs = {b_id: [] for b_id in self.enum}
        for data in Map:
            for b_i, b_name in enumerate(batchXs):
                bX = Xs[data][np.array(Bs[data][:, b_i], dtype=bool)]
                if len(bX):
                    batchXs[b_name].append(bX)
        for b_id in batchXs:
            if len(batchXs[b_id]):
                b_vals = np.concatenate(batchXs[b_id])
                self.stats[b_id] = {'mean': np.mean(b_vals, axis=0),
                                    'sd': np.std(b_vals, axis=0)}

    def standardizeData(self):
        """
        Applies standardization to the data based on the previously calculated batch-wise
        statistics. The standardized data is then returned by each 'Data' constraint instead of the original data.
        """
        allBs = self.Ys(self._inds, called=True)
        Map = self.dataMap(self._inds)
        for data in allBs:
            data.normed = np.copy(data.X)
            d_batch = np.array(allBs[data])
            d_inds = Map[data]
            for b_name in self.stats:
                b_mask = d_batch == b_name
                if np.sum(b_mask):
                    d_b_inds = d_inds[b_mask]
                    data.normed[d_b_inds] = (data.X[d_b_inds] - self.stats[b_name]['mean']) / self.stats[b_name]['sd']


class MMD(Normalization):
    """
    The MMD (Maximum Mean Discrepancy) class which extends the Normalization class.
    This class is responsible for calculating a statistical test for the difference
    in distributions between two groups, then using a loss to minimise that difference.
    As a result, specified groups become aligned in the latent representation.

    Attributes:
    -----------
    pull: float
        Strength of the MMD loss which converges the encoders.
    kernel: function, optional
        Kernel function to compute the MMD.
    tile: bool
        Determines if all pairwise sample comparisons should be included in MMD calculation.
    """
    def __init__(self, trainEmbedding=True, pull=1.0, **kwargs):
        super(MMD, self).__init__(trainEmbedding=trainEmbedding, **kwargs)
        self._trainYs = None
        self._valYs = None
        self.pull = pull
        self.kernel = None
        self._kernelFunc = None
        self.tile = False

    def archive(self):
        d = super().archive()
        d['pull'] = self.pull
        return d

    def unarchive(self, d):
        super().unarchive(d)
        self.pull = d['pull']

    def hyperparams(self):
        h = super().hyperparams()
        h['frequency'] = self.frequency
        return h

    def index(self):
        super(MMD, self).index()
        if self._inds is not None:
            self.getIndexLabels()
        self._kernelFunc = None

    def getIndexLabels(self):
        """
        Retrieves condition labels corresponding to the indexed data and stores them for faster sampling.
        """
        inds = self.inds(validation=False)
        self._trainYs = self.stack(self.Ys(inds, called=True))
        inds_val = self.inds(validation=True)
        if len(inds_val):
            self._valYs = self.stack(self.Ys(inds_val, called=True))

    def balance(self, prediction=None, prop=1.0):
        super(Classification, self).balance(prediction, prop=prop)
        self._resampled = np.sort(self._resampled)
        self.getIndexLabels()

    def batch(self, bs, validation=False):
        """
        Selects random samples from two random conditions.
        """
        inds = self.inds(validation=validation)
        conditions = [self.enum[n] for n in np.random.permutation(len(self.enum))[0:2]]
        ys = self._trainYs if not validation else self._valYs
        c1_inds = np.arange(len(inds))[ys == conditions[0]]
        c2_inds = np.arange(len(inds))[ys == conditions[1]]
        s_per_cond = int(np.floor(bs/2))
        c1_batch_inds = np.random.permutation(c1_inds)[:s_per_cond]
        c2_batch_inds = np.random.permutation(c2_inds)[:s_per_cond]
        selInds = np.concatenate([inds[c1_batch_inds], inds[c2_batch_inds]])
        return selInds

    def cartesianProduct(self, x, y):
        """
        Expands the data to contain all pairwise comparisons between datapoints.
        """
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
        tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
        return tiled_x, tiled_y

    def multiscaleGaussian(self, x, y):
        """
        Computes a multiscale Gaussian kernel between x and y.

        Parameters:
        -----------
        x, y: array-like
            Input arrays.

        Returns:
        --------
        array-like:
            Kernel values.
        """
        if self.tile:
            x, y = self.cartesianProduct(x, y)
        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
        def squared_distance(x, y):  # returns the pairwise euclidean distance
            r = K.expand_dims(x, axis=1)
            return K.sum(K.square(r - y), axis=-1)
        beta = 1. / (2. * (K.expand_dims(sigmas, 1)))
        distances = squared_distance(x, y)
        s = K.dot(beta, K.reshape(distances, (1, -1)))
        return K.reshape(tf.reduce_sum(input_tensor=tf.exp(-s), axis=0), K.shape(distances)) / len(sigmas)

    def mmd2(self, x, y):
        """
        Computes the MMD squared loss between x and y.

        Parameters:
        -----------
        x, y: array-like
            Input arrays.

        Returns:
        --------
        float:
            MMD squared value.
        """
        return K.mean(self.multiscaleGaussian(x, x)) \
               + K.mean(self.multiscaleGaussian(y, y)) \
               - (2*K.mean(self.multiscaleGaussian(x, y)))

    def forward(self, inds):
        """
        Computes the forward pass, calculating the MMD loss.

        Parameters:
        -----------
        inds: array-like
            Indices of the samples.

        Returns:
        --------
        tuple:
            Dictionary of losses and weighed loss value.
        """
        Map = self.dataMap(inds)
        Zs = self.getInput(Map, mean=True)
        Z_cat = tf.concat([Zs[data] for data in Map], axis=0)
        Ys = self.Ys(inds, called=True)
        Y_cat = tf.concat([Ys[data] for data in Map], axis=0)
        conds, idx = tf.unique(Y_cat)
        if UVAE_DEBUG:
            if len(conds) > 2:
                print('Warning: more than two conditions were sampled in batch:', self.name)
        c1_mask = idx == 0
        c2_mask = idx == 1
        z1 = Z_cat[c1_mask]
        z2 = Z_cat[c2_mask]
        if self.kernel is not None:
            if self._kernelFunc is None and self.kernel.func is not None:
                self._kernelFunc = keras.Model(self.kernel.func.layers[0].input, self.kernel.func.layers[-1].layers[-2].output)
            z1 = self._kernelFunc(z1)
            z2 = self._kernelFunc(z2)
        dist = self.mmd2(z1, z2)
        return {self.name: dist * self.pull}, dist * self.weight


class History:
    """
    Maintains a record of training metrics and provides utilities for early stopping.

    The History class tracks the performance metrics over training epochs. It also includes functionality to monitor
    a given validation metric for early stopping, either based on lack of improvement or when the training loss
    reaches a threshold.

    Attributes
    ----------
    epoch : int
        Current epoch number.
    earlyStop : int
        Number of epochs with no improvement in the specified key metric after which training should be stopped.
    earlyStopKey : str
        The key metric to monitor for early stopping.
    stopLoss : float
        Threshold for the training loss; training stops when the loss goes below this value.
    shouldStop : bool
        Flag indicating if the training should stop based on the provided criteria.
    noImprovement : int
        Counter for the number of epochs without improvement in the specified key metric.
    minValLoss : float
        Minimum validation loss observed so far.
    improved : bool
        Flag indicating if the specified key metric has improved in the current epoch.
    history : dict
        Dictionary to store the accumulated values for each metric across epochs.
    accum : dict
        Temporary storage to accumulate values within an epoch.
    timers : dict
        Dictionary to track time-related information for various processes.

    Methods
    -------
    append(key: str, value: float):
        Append a value to the accumulator for a given metric key.
    change(key: str, change: float):
        Change the last value in the accumulator for a given metric by a specified amount.
    accumulate(sum=True):
        Calculate the sum or mean of the accumulated values for each metric, add to the history, and check stopping criteria.
    time(name: str, reset=True) -> float:
        Record or retrieve the time for a specified process.
    print(s='') -> str:
        Print a summary of the metrics for the current epoch.

    Notes
    -----
    This class is designed to be used alongside training loops to keep track of performance metrics and potentially
    stop training early based on specified criteria.
    """
    def __init__(self, earlyStop=0, stopLoss=0, earlyStopKey='val_loss'):
        self.epoch = 0
        self.earlyStop = earlyStop # stop based on key loss not improving
        self.earlyStopKey = earlyStopKey
        self.stopLoss = stopLoss # or stop based on training loss reaching a value
        self.shouldStop = False
        self.noImprovement = 0
        self.minValLoss = float('inf')
        self.improved = False
        self.history = {}
        self.accum = {}
        self.timers = {}
        self.time('epoch')

    def append(self, key, value):
        if not key in self.accum:
            self.accum[key] = []
        self.accum[key].append(value)

    def change(self, key, change):
        if not key in self.accum:
            self.accum[key] = [0]
        last = self.accum[key][-1]
        new = last + change
        self.accum[key].append(new)

    def accumulate(self, sum=True):
        for key in self.accum.keys():
            if len(self.accum[key]):
                if sum:
                    value = np.nansum(self.accum[key])
                else:
                    value = np.nanmean(self.accum[key])
                self.accum[key] = []
                if not key in self.history:
                    self.history[key] = []
                self.history[key].append(value)
                if key == self.earlyStopKey:
                    if value < self.minValLoss:
                        self.minValLoss = value
                        self.noImprovement = 0
                        self.improved = True
                    else:
                        self.noImprovement += 1
                        self.improved = False
                        if self.noImprovement >= self.earlyStop and self.earlyStop > 0:
                            self.shouldStop = True
                elif key == 'loss':
                    if value < self.stopLoss and self.stopLoss > 0:
                        self.shouldStop = True

    def time(self, name, reset=True):
        if name not in self.timers:
            self.timers[name] = time.time()
            return 0.0
        else:
            t = time.time()
            d = t - self.timers[name]
            if reset:
                self.timers[name] = t
            return d

    def print(self, s=''):
        s += 'Epoch {} ({}s): '.format(int(self.epoch), int(self.time('epoch')))
        if 'loss' in self.history:
            s += 'loss: {:.4f}'.format(self.history['loss'][-1])
        if 'val_loss' in self.history:
            s += '  val_loss: {:.4f}'.format(self.history['val_loss'][-1])
        for k in sorted(list(self.history.keys())):
            if k not in ['loss', 'val_loss']:
                s += '\t{}: {:.4f}'.format(k, self.history[k][-1])
        print(s)
        return s


class LisiValidationSet:
    """
    A utility class for LISI (Local Inverse Simpson's Index) based model selection.

    This class is used to reference a fixed set of data for LISI calculation, and optionally
    updating class predictions.

    Attributes
    ----------
    dm : DataMap
        The data map object referencing the samples to validate.
    normClasses : bool
        Flag to determine if the classes should be normalized between batches.
    perplexity : int
        The perplexity value used for LISI calculation, which determines the size of local neighborhoods.
    batchWeight : float
        Weight for the batch constraint in LISI calculation.
    labelWeight : float
        Weight for the label constraint in LISI calculation.
    batchConstraint : Labeling
        Labeling constraint containing batch assignments.
    labelConstraint : Labeling
        Labeling constraint for class labelling (or a classifier making dynamic class predictions).
    batchRange : tuple, optional
        Range for batch constraint, defaults to the range of unique batches.
    labelRange : tuple, optional
        Range for label constraint, defaults to the range of unique labels.
    batch : array-like
        Batch labels as predicted from the batch constraint.
    labeling : array-like
        Class labels as predicted from the label constraint.

    Methods
    -------
    update():
        Update the batch and label arrays using the provided constraints.

    Notes
    -----
    This class is designed to store a fixed sample subset during model selection.
    """
    def __init__(self, dm:DataMap,
                 batchConstraint:Labeling,
                 labelConstraint:Labeling,
                 normClasses=False,
                 perplexity=30,
                 batchRange=None,
                 labelRange=None,
                 batchWeight=1.0,
                 labelWeight=1.0):
        self.dm = dm
        self.normClasses = normClasses
        self.perplexity = perplexity
        self.batchWeight = batchWeight
        self.labelWeight = labelWeight
        self.batchConstraint = batchConstraint
        self.labelConstraint = labelConstraint
        self.batchRange = batchRange
        self.labelRange = labelRange
        self.batch = None
        self.labeling = None

    def update(self):
        self.batch = self.batchConstraint.predictMap(self.dm, stacked=True)
        self.labeling = self.labelConstraint.predictMap(self.dm, stacked=True)
        if self.batchRange is None:
            self.batchRange = (1, len(set(self.batch)))
        if self.labelRange is None:
            self.labelRange = (1, len(set(self.labeling)))


class ModelSelectionHistory:
    """
    Stores the history of results during the model selection process and ensures
    the desired number of iterations has been evaluated.

    This class maintains a history of past results and a set of current results.
    The history can be compounded to group sets of results,
    e.g., after finishing a hyperparameter search round.

    Attributes
    ----------
    source : UVAE
        The optimised UVAE model.
    targetIterations : int
        Total number of intended iterations or configurations to be evaluated.
    currentResults : list
        List of results for the ongoing round of evaluations.
    pastResults : list of lists
        Nested list where each sublist corresponds to a set of results from past rounds.

    Methods
    -------
    results():
        Returns a flattened list of all results, both current and past.
    __len__():
        Returns the total number of results recorded so far.
    addIterations(i: int):
        Increments the target number of iterations by `i` and returns the number of results still needed.
    compound():
        Moves current results to the past results list and resets current results.
    addResult(arr: list):
        Adds a list of results to the current results.
    """
    def __init__(self, source):
        self.source = source
        self.targetIterations = 0
        self.currentResults = []
        self.pastResults = []

    def results(self):
        res = []
        res.extend(self.currentResults)
        for r in self.pastResults:
            res.extend(r)
        return res

    def __len__(self):
        return int(np.sum([len(r) for r in self.pastResults]) + len(self.currentResults))

    def addIterations(self, i):
        self.targetIterations += i
        return int(self.targetIterations - len(self))

    def compound(self):
        if len(self.currentResults):
            self.pastResults.append(self.currentResults)
            self.currentResults = []

    def addResult(self, arr):
        self.currentResults.append(arr)
