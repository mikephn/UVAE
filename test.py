from src.tools import *

# download sample data: https://zenodo.org/record/5748302/files/ToyData-3x3.pkl?download=1

toyDs = unpickle('ToyData-3x3.pkl')

X0, X1, X2 = toyDs['Xs']
chs0, chs1, chs2 = toyDs['enum']['markers']

Y0, Y1, Y2 = toyDs['Ys']['label']

B0, B1, B2 = toyDs['Ys']['batch']

from src.UVAE import *

uv = UVAE('toy.uv')

p0 = uv + Data(X0, channels=chs0, name='Panel 0')
p1 = uv + Data(X1, channels=chs1, name='Panel 1')
p2 = uv + Data(X2, channels=chs2, name='Panel 2')

batch = uv + Labeling(Y={p0: B0, p1: B1, p2: B2}, name='Batch')

ae0 = uv + Autoencoder(name=p0.name, masks=p0, conditions=[batch])
ae1 = uv + Autoencoder(name=p1.name, masks=p1, conditions=[batch])
ae2 = uv + Autoencoder(name=p2.name, masks=p2, conditions=[batch])

mmd = uv + MMD(Y=batch.Y, name='MMD', pull=10)

ctype = uv + Classification(Y={p0: Y0, p1: Y1, p2: Y2}, nullLabel='unk', name='Cell type')
ctype.resample(mmd)

red2d = uv + Projection(latent_dim=2, name='2D')

from src.UVAE_diag import plotsCallback

uv.train(20, callback=plotsCallback)

plotsCallback(uv, doUmap=True, outFolder=ensureFolder('umap/'))
