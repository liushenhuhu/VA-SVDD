from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
def tsne(latent):
    pos = TSNE(n_components=2).fit_transform(latent)

    return pos
def pca(latent):
    pca = PCA(n_components=2)
    pos = pca.fit_transform(latent.cpu())

    return pos


def Umap(latent):
    reducer = umap.UMAP(n_components=2)
    pos = reducer.fit_transform(latent.cpu())
    return pos


def isomap(latent):
    iso = Isomap(n_components=2)
    pos = iso.fit_transform(latent.cpu())

    return pos


def mds(latent,):
    mds = MDS(n_components=2)
    pos = mds.fit_transform(latent.cpu())

    return pos
