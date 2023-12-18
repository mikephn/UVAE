## UVAE Quick Start

1. [Introduction](#1-introduction)
   - [Dependencies](#11-dependencies)
2. [Usage](#2-usage)
   - [Creating Semi-Supervised Models](#21-creating-semi-supervised-models)
     - [Instantiation or Loading of Models](#211-instantiation-or-loading-of-models)
     - [Adding Data](#212-adding-data)
     - [Adding Labelling](#213-adding-labelling)
     - [Adding Classifiers](#214-adding-classifiers)
     - [Visualising Latent Space](#215-visualising-latent-space)
     - [Training](#216-training)
   - [Alignment of Disjoint Data](#22-alignment-of-disjoint-data)
     - [Merging by Shared Features](#221-merging-by-shared-features)
     - [Merging by Density Matching](#222-merging-by-density-matching)
   - [Correcting Batch Effects](#23-correcting-batch-effects)
     - [Latent Space Normalisation](#231-latent-space-normalisation)
     - [Conditional Autoencoding](#232-conditional-autoencoding)
   - [Resampling to Match Class Distributions](#24-resampling-to-match-class-distributions)
   - [Conditional Generation of Data](#25-conditional-generation-of-data)
   - [Hyper-Parameter Tuning](#26-hyper-parameter-tuning)
3. [Summary](#3-summary)

## 1 Introduction

UVAE is a deep learning framework for training autoencoder-based latent variable models through simple description of modelling objectives.

You can create models to automatically integrate disjoint data streams (characterised by different feature sets), correct batch effects, perform regression, classification, and imputation over the joint latent space.

All normalisation and merging is automatically class-balanced to avoid distortion and over-alignment.

Currently column data is supported.

#### 1.1 Dependencies

`tensorflow==2.10.1 scipy` 

Optional (Python):

`arm-mango` for hyper-parameter tuning,

`harmonypy` for LISI metric calculation,

`sklearn` (Gaussian Mixture) or `scanpy` (Leiden) for clustering,

`matplotlib umap-learn` for visualisation.


Optional (R):

`immunogenomics/lisi` for LISI metric calculation (if harmonypy is not used),

`cyCombine` for EMD/MAD metric calculation.

## 2 Usage

To use throughout this tutorial as an example, let's import a toy dataset containing samples from a flow cytometry experiment. This sample was split into 3 separate panels (using different sets of features). Each panel was further split into 3 batches. All splits were performed to result in random uneven class distribution across batches. Each batch was re-normalised to simulate batch effects. Finally, 90% of the ground truth labeling was hidden.
```python
from src.tools import *

# download sample data: https://zenodo.org/record/5748302/files/ToyData-3x3.pkl?download=1
toyDs = unpickle('ToyData-3x3.pkl')

X0, X1, X2 = toyDs['Xs'] # input values

chs0, chs1, chs2 = toyDs['enum']['markers'] # feature names (flow markers)

Y0, Y1, Y2 = toyDs['Ys']['label'] # cell-type labels

B0, B1, B2 = toyDs['Ys']['batch'] # batch assignment
```

    >>> X0.shape
    (27903, 19)
    >>> X1.shape
    (33176, 19)
    >>> X2.shape
    (38921, 19)
    
    >>> chs0
    ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W', 'CD56', 'HLA DR', 'CD11c', 'CD14', 'CD16', 'CD45', 'CD11b', 'CD3', 'CD62L', 'CD123', 'LD', 'CD10', 'CD24']
    >>> chs1
    ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W', 'CD56', 'HLA DR', 'CD11c', 'CD14', 'CD16', 'CD45', 'CD11b', 'CD3', 'CD62L', 'CD123', 'LD', 'CD1c', 'CD19']
    >>> chs2
    ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W', 'CD56', 'HLA DR', 'CD11c', 'CD14', 'CD16', 'CD45', 'CD11b', 'CD3', 'CD62L', 'CD10', 'CD24', 'CD1c', 'CD19']

    >>> np.column_stack(np.unique(Y0, return_counts=True))
    array([['B cells', '24'],
           ['Basophils', '59'],
           ['DC cells', '5'],
           ['Debris', '573'],
           ['Doublets', '231'],
           ['Eosinophils', '52'],
           ['Monocytes', '113'],
           ['NK cells', '98'],
           ['Neutrophils', '627'],
           ['T cells', '574'],
           ['unk', '25547']], dtype='<U21')

    >>> np.column_stack(np.unique(B0, return_counts=True))
    array([['p0b0', '9979'],
           ['p0b1', '8659'],
           ['p0b2', '9265']], dtype='<U21')

Notice that because some channels are different in each panel we can't simply concatenate the data and make a single embedding. To avoid losing unique information in each panel we will make independent latent embeddings and join them during training with soft loss constraints.

### 2.1 Creating semi-supervised models

#### 2.1.1 Instantiation or loading of models

Create a model by specifying a path where the parameters should be saved. If a model already exists at this path, it is loaded during each consecutive run.

```python
from src.UVAE import *

uv = UVAE('toy.uv')
```

#### 2.1.2 Adding data
To make an unsupervised embedding for a data stream first create a UVAE object with a saving path, then add the input data, retaining a reference to each stream as a *Data* object:

```python
uv = ... # created previously

p0 = uv + Data(X0, channels=chs0, name='Panel 0')
p1 = uv + Data(X1, channels=chs1, name='Panel 1')
p2 = uv + Data(X2, channels=chs2, name='Panel 2')
```

The *p0 ... p2* objects are constraints referencing the added data. We will use them to add labelling and merging constraints on top.

By default, for each added *Data* stream, UVAE will create a variational autoencoder, which will be used to embed those data points for any further constraint. Default embedding is to **50** dimensions. See *UVAE_hyper.py* for other hyper-parameters.

#### 2.1.3 Adding labelling

You can add a known ground-truth labelling to data. For example, batch assignment is typically fully known and fixed throughout training:

```python
batch = uv + Labeling(Y={p0: B0, p1: B1, p2: B2}, name='Batch')
``` 

#### 2.1.4 Adding classifiers

For partially known assignments, classifiers can be trained to extrapolate a partial labelling to the whole dataset. In this example, training labels are stored in *Y0..2* corresponding to each *X0..2*, and *unk* is used to indicate a lack of annotation.

```python
ctype = uv + Classification(Y={p0: Y0, p1: Y1, p2: Y2}, nullLabel='unk', name='Cell type')
```

The above lines will add a classifier network predicting from the shared latent space, which can be used to infer cell-types across all samples.

#### 2.1.5 Visualising latent space

To be able to visualise the high-dimensional space during training we can add a consecutive autoencoder projecting the latent space to 2D, which is trained in parallel with the primary autoencoders:

```python
red2d = uv + Projection(masks=[p0, p1, p2], latent_dim=2, name='2D')
```

#### 2.1.6 Training

Training is performed by calling *train*. Model is saved automatically to the path specified during initialisation. By default, only previously untrained constraints are updated to enable sequential model training. Optionally, a callback function can be used to visualise the progress every epoch:

```python
from src.UVAE_diag import plotsCallback

uv.train(20, callback=plotsCallback)
```

![Training of 3 modalities in parallel.](https://i.imgur.com/xqpEtHV.gif)

Notice that at the moment the embeddings of the three panels are not aligned. This is because no merging constraints were added.

### 2.2 Alignment of disjoint data

#### 2.2.1 Merging by shared features

We can use the fact that our 3 panels share a large number of markers. A *Subspace* constraint creates an additional autoencoder which uses all shared markers as input, and adds a pairwise merging loss between all included modalities and itself.

```python
sub = uv + Subspace(masks=[p0, p1, p2], name='Shared markers', pull=1)
```
![Training of 3 modalities with shared subspace merging.](https://i.imgur.com/0jnkdZb.gif)

This method by itself is not sensitive to class imbalance across merged data, because it creates synthetic pairs for each individual sample. It is, however, sensitive to batch effects present across the data. [Latent space normalisation](#231-latent-space-normalisation) in conjunction with [conditional autoencoding](#232-conditional-autoencoding) can be used to address this.

#### 2.2.2 Merging by density matching

*Maximum Mean Discrepancy* constraint can be used for merging data without requiring shared channels. Instead, it assumes that the latent distribution between merged datasets has a matching density. UVAE implements MMD with multiscale RBF kernel from [trVAE](https://doi.org/10.1093/bioinformatics/btaa800). If MMD is defined between more than two sets of data, it is applied between two random sets in each training minibatch. Let's add MMD to merge the 3 panels:

```python
p_ids = {p0: np.repeat('0', len(p0.X)),
            p1: np.repeat('1', len(p1.X)),
            p2: np.repeat('2', len(p2.X))}

mmd = uv + MMD(Y=p_ids, name='MMD', pull=1)
```
![Training of 3 modalities with MMD.](https://i.imgur.com/FZGQ2RF.gif)

We can adjust the strength of merging for each constraint by increasing *pull*. However, remember that MMD forces latent densities to match. We know that our panels contain different proportions of cell-types, and therefore shouldn't match exactly. UVAE can address this problem by performing automatic [control set balancing](#24-resampling-to-match-class-distributions).

### 2.3 Correcting batch effects

To correct batch effects we need to supply a batch assignment, which defines the groups that should be matched. We will use the *Labeling* added in the [previous section](#213-adding-labelling).

#### 2.3.1 Latent space normalisation

UVAE is designed to perform running normalisation of constraints based on dynamically sampled control sets. *Normalization* implements latent space arithmetics that mean-centers the embeddings of batches in the latent space, optionally to a specified target:

```python
ln = uv + Normalization(Y=batch.Y, name='Latent norm', target='p0b0')
```

Specifying this constraint will automatically result in the embeddings passed to any follow-up constraints (such as *Classification* or *Subspace*) to be normalised between batches. This normalisation is applied gradually, based on the *ease_epochs* hyper-parameter.

Notice that because our batches contain imbalanced proportions of cell types, the mean statistic will not be directly comparable. To sample a similar control set from each batch before calculating the statistic, this constraint should be used in conjunction with [resampling](#24-resampling-to-match-class-distributions).

#### 2.3.2 Conditional autoencoding

Let's assume we want to create an autoencoder for data which consists of 3 batches, and we want to apply MMD to merge the batches in the latent space:

```python
uv = UVAE('mmd.uv')
p0 = uv + Data(X0, name='Panel')
mmd = uv + MMD(Y={p0: B0}, name='MMD_batch')
```

For training, UVAE will automatically add an autoencoder for the *Data* object and train to minimise a combination of reconstruction error and MMD. This however poses contradictory objectives: we want the latent distributions of batches to be indistinguishable, but at the same time we want the reconstruction of each batch to be faithful (and therefore contain batch-specific information).

We can address this problem by conditioning the autoencoder to provide it with batch identifying information, so that it doesn't need to be redundantly captured in the latent space.

To add conditioning we need to manually instantiate the autoencoder and specify one or more conditions:

```python
uv = UVAE('mmd.uv')
p0 = uv + Data(X0, name='Panel')
batch = uv + Labeling(Y={p0: B0}, name='Batches')
ae0 = uv + Autoencoder(name=p0.name, masks=p0, conditions=[batch], condEncoder=True)
mmd = uv + MMD(Y={p0: B0}, name='MMD_batch')
```

Any autoencoder (for example, *Subspace*) can be made conditional in this way. More than one conditioning can be specified simultaneously, each containing multiple classes. By default, both encoder and decoder are conditioned.

### 2.4 Resampling to match class distributions

Resampling is a core feature in the UVAE framework. Any *Labeling* or *Classification* constraint can be used dynamically during model training to balance the contents of samples used to train any other constraint.
This is particularly useful to make sure that the samples used for [latent space normalisation](#231-latent-space-normalisation) or [aligning by density matching](#222-merging-by-density-matching) contain the same class distribution across batches.

```python
ctype = uv['Cell type'] # cell-type classifier
ln = uv['Latent norm'] # target for resampling
ctype.resample(ln)
```

The above code will result in the cell-type classifier predicting the classes of all the samples affected by *Normalization* at the end of every epoch, and balancing them to have the same number of cells from each class in each batch.


If the target of rebalancing is an *Autoencoder* or *Regression* (including their subclasses: *Classification*, *Projection*, *Subspace* etc.) it results in taking equal number of samples from each source class during training. If the target is a *Normalization* subclass (including *MMD*), the resampling is performed **per batch**. In this case, additional *balanceBatchAvg* parameter can be set on the target constraint:

```python
ln = uv + Normalization(Y=batch.Y, name='Latent norm', target='p0b0', balanceBatchAvg=True)
ctype.resample(ln)
```

When *balanceBatchAvg* is set to *True*, it makes resampling first determine the proportions of classes across all batches, then resample to a mean proportion. If set to *False*, each class is sampled equally (not preserving global proportions). When classes are not shared across batches (for example, when independent clustering is used), this parameter should be set to *False*.

More than one resampling can be added to each target, in which case the underlying data will be divided into equal parts for each source. Additionally, the resampling and normalisation constraints can be eased in, so that classifiers have time to train. This is determined by *ease_epochs* parameter in UVAE_hyper.py.

### 2.5 Conditional generation of data

The **reconstruct()** function has two modes of operation. If no *channels* are specified, each sample from the requested data map is generated from the corresponding autoencoder. If *channels* are specified, each sample is first encoded by its panel encoder, then generated from all the decoders which support requested channels, and averaged. This allows for cross-modal generation and imputation of missing features.

E.g. to impute all available markers for all samples using all available panel decoders:

```python
# map covering all data
all_samples = uv.allDataMap()
# combined markers from all panels
all_markers = np.unique(np.concatenate([d.channels for d in all_samples]))
# generate all markers for every sample by averaging decoders across panels
rec_imputed = uv.reconstruct(all_samples, channels=all_markers)
```

A subset of panels which we want to use to decode can be explicitly specified:

```python
rec_imputed = uv.reconstruct(all_samples, channels=all_markers, decoderPanels=[p0, p1])
```

Specific target conditions can be used to generate data for each conditional autoencoder, or normalisation constraint:

```python
targets = {ln: 'p0b0', batch: 'p0b0'}
rec_imputed = uv.reconstruct(all_samples, channels=all_markers, targets=targets)
```

### 2.6 Hyper-parameter tuning

Hyper-parameters such as model size, learning rates (separate for *unsupervised*, *supervised*, and *merge* losses), mutual frequency of training of each constraint, as well as *pull* strength can be automatically determined using *mango* Bayesian optimiser. The ranges and default values can be adjusted in *UVAE_hyper.py*. Each constraint has a *weight* parameter, which can be used to adjust its importance for model selection only. Optimisation is done by calling *optimize* on the instantiated UVAE object after adding all required data and constraints:

```python
uv.optimize(iterations=20, # training attempts
            maxEpochs=30, # max epochs during each training run
            earlyStopEpochs=0, # 0 if early stopping should not be used
            samplesPerEpoch=100000, # data samples per epoch
            valSamplesPerEpoch=100000, # validation samples per epoch
            sizeSeparately=True, # optimise width and depth of each constraint separately
            subset=['latent_dim', 'width', 'pull-MMD_batch'], # a subset of hyper-parameters to be optimised
            callback=None) # callback function called with the resulting model after each optimisation attempt
```

*Subset* is a list of hyper-parameter names. To include *pull* or *frequency* of individual constraints, use *pull-name* or *frequency-name* with the constraint name.

LISI metric can be included in the hyper-parameter optimisation loss. First, create an object storing information for LISI calculation:

```python
lisiSet = LisiValidationSet(dm=uv.allDataMap(), # data range to calculate LISI over
                            labelConstraint=ctype, # Labeling or Classification defining classes (which should remain separated)
                            batchConstraint=batch, # Labeling or Classification defining batches (which should be merged)
                            normClasses=True, # should equal number of class instances be sampled from each batch
                            labelRange=(1.0, 10.0), # expected range of score for class labeling (defaults to min:1.0, max:number of classes)
                            labelWeight=1.0, # label score contribution
                            batchRange=(1.0, 10.0), # expected range of score for batch labeling (defaults to min:1.0, max:number of batches)
                            batchWeight=1.0, # batch score contribution
                            perplexity=100) # LISI perplexity parameter
```

Any other metric can be added as a hyper-parameter optimisation loss by specifying a function which is called after each optimisation run. This function should accept the newly trained model as input, and return the loss contribution (lower is better):

```python
def customHyperoptLoss(model):
    score = ...
    return -score

uv.optimize(50, customLoss=customHyperoptLoss)
```

## 3 Summary

To summarise, let's create a model which merges the panels using shared channels and MMD, while performing batch effect correction with both conditional autoencoding and latent space normalisation. The classifier will be used to predict cell-types across the dataset and balance the sensitive merging and normalisation constraints:

```python
from src.UVAE import *

# Instantiate the model
uv = UVAE('toy.uv') 

# Add data series with disparate features
p0 = uv + Data(X0, channels=chs0, name='Panel 0')
p1 = uv + Data(X1, channels=chs1, name='Panel 1')
p2 = uv + Data(X2, channels=chs2, name='Panel 2')

# Add batch labelling
batch = uv + Labeling(Y={p0: B0, p1: B1, p2: B2}, name='Batch')

# Manually create autoencoders to enable conditioning with batch annotation
ae0 = uv + Autoencoder(name=p0.name, masks=p0, conditions=[batch])
ae1 = uv + Autoencoder(name=p1.name, masks=p1, conditions=[batch])
ae2 = uv + Autoencoder(name=p2.name, masks=p2, conditions=[batch])

# Add latent space normalisation between batches
ln = uv + Normalization(Y=batch.Y, name='Latent norm')

# Add panel merging by using shared features and make the Subspace autoencoder conditional
sub = uv + Subspace(masks=[p0, p1, p2], name='Shared markers', conditions=[batch], pull=1)

# Add batch merging by using MMD
mmd = uv + MMD(Y=batch.Y, name='MMD', pull=1)

# Add a cell-type classifier with 'unk' set as null token
ctype = uv + Classification(Y={p0: Y0, p1: Y1, p2: Y2}, nullLabel='unk', name='Cell type')

# Use the classifier to class-balance the samples used for latent normalisation and MMD
ctype.resample(ln)
ctype.resample(mmd)

# Add a 2D projection of the latent space for visualisation purposes
red2d = uv + Projection(latent_dim=2, name='2D')

# Train the model
uv.train(20)
```

![Training of 3 modalities with MMD and resampling.](https://i.imgur.com/J1Cf4LW.gif)