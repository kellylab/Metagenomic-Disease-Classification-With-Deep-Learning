import pandas as pd
import numpy as np
import os
import pickle
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
matplotlib.interactive(True)

# Load data

def clf(df):
    """ Computes centered log-ratio transform for each row of dataframe : x' = log(x)/g, where g = geometric mean of row. """
    for i in range(len(df.index)):
        row = df.iloc[i]
        g = np.exp(
            np.mean(
            np.log(
            row.replace(0,1)
            )))
        df.iloc[i] = np.log(df.iloc[0]/g).replace(np.float64('-inf'), 0)

    return df

def compute_pca(df):
    """ Computes PCA of dataframe. """
    print("Ready to rrrrrummmblllleeeee")
    pca = IncrementalPCA(n_components=5, batch_size=10)
    pca.fit(df)
    return pca

def compute_tsne(df):
    """ Computes tsne of dataframe. """
    print("Hammer time")
    tsne = TSNE(n_components=2)
    transforms = tsne.fit_transform(df)
    return transforms

def convert_tsne(transforms, dataframes, key_sets, key):
    """ Breaks down transformed data and labels it. """
    start = 0
    end = 0
    labelled_transforms = {}
    for df, name in zip(dataframes[key], key_sets):
        end = end + np.shape(df)[0]
        labelled_transforms[name] = transforms[start:end]
        start = end

    return labelled_transforms

def _fit_model(model, dataframes, key, key_sets=None):
    """ Fit data in dataframes to scikitlearn model """
    if key_sets is None:
        key_sets = []
        for dataset in datasets:
            if key in os.listdir(os.path.join(dataset_dir, dataset)):
                key_sets.append(dataset)

    transforms = {}
    i = 0
    for dataset in key_sets:
        transforms[dataset] = model.transform(dataframes[key][i])
        i = i + 1
    return transforms

def fit_model(model, df_dict):

    transforms = {}
    for key in df_dict:
        transforms[key] = model.transform(df_dict[key])

    return transforms

colors = lambda :cm.rainbow(np.random.rand(22))

def plot_transforms(transforms, title="", save_file = None, xlabel = '', ylabel = ''):
    """ transforms is a dictionary mapping dataset names to transformed values. Title is title of plot. """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = len(transforms.keys())
    plots = []
    for color, key in zip(colors(), transforms):
        plots.append(ax.scatter(transforms[key][:,0], transforms[key][:,1], c=color, cmap='Accent'))
    ax.legend(plots, transforms.keys())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if save_file:
        plt.savefig(save_file+".svg", format='svg', dpi=1200)
    return ax.get_xlim(), ax.get_ylim()

def plot_sequential_transforms(transforms, title="", save_file=None, xlabel = '', ylabel  ='', xlim=None, ylim=None):
    """ Plots each element of transforms on a separate plot. """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = len(transforms.keys())
    colors = cm.rainbow(np.linspace(0,1,n))
    for color, key in zip(colors(), transforms):
        ax.clear()
        ax.scatter(transforms[key][:,0], transforms[key][:,1], c=color, cmap='Accent')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title+" " +key)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if save_file is not None:
            plt.savefig(save_file+"_"+key+".svg", )

def plot_transforms_3d(transforms, colors=None, show = False, fig=None):
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if colors == None:
        colors = cm.rainbow(np.random.rand(22))
    plots = {}
    for color, key in zip(colors(), transforms):
        ax.scatter(transforms[key][:,0], transforms[key][:,1], transforms[key][:,2], c=color).set_legend(key)
    if show:
        plt.show()
