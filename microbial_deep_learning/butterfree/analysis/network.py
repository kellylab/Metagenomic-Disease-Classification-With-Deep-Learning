from butterfree.data import loader 
from collections import defaultdict 
import networkx as nx 
import pygraphviz as pgv 
import matplotlib.pyplot as plt 
import pandas as pd
import os 
import re
import numpy as np
import torch
from scipy import linalg
from networkx.drawing.nx_agraph import write_dot, graphviz_layout 

cols = loader.all_columns_metaphlan() 

def coo(A): 
    rows = [] 
    columns = [] 
    values = [] 
    for i in range(A.shape[0]): 
        for j in range(A.shape[1]): 
            if A[i,j]: 
                rows.append(i) 
                columns.append(j) 
                values.append(A[i,j]) 
    return rows, columns, values

def coo_tensor(A):

    rows, columns, values = coo(A)
    return torch.tensor([rows, columns], dtype=torch.long)

def get_eigenvectors(taxonomies):
    """
    Returns eigenvectors and eigenvalues of Graph Laplacian.
    """
    G = phylogenetic_tree(taxonomies)
    G2 = G.to_undirected()
    A = nx.convert_matrix.to_numpy_matrix(G2)    
    # Align nodes with taxonomic names
    nodes = [x for x in G2.nodes()]
    indices = reorder_columns(nodes, taxonomies)
    A = A[indices, :][:, indices] # Shuffle rows and columns of adgacency matrix
    # Compute laplacian
    L = nx.laplacian_matrix(G2)
    L = L[indices, :][:, indices] # Shuffle rows and columns of laplacian matrix
    L = L.toarray()
    # Compute eigenvectors and eigenvalues 
    W, V = linalg.eigh(L)

    return A, L, W, V

def reorder_columns(nodes, taxonomies):
    """
    Returns a list of indices to use to shuffle the elements of nodes such that shuffled(nodes) == taxonomies
    """
    taxa = [t.split('|')[-1] for t in cols]
    if set(nodes) != set(taxa):
        raise ValueError("Elements of both lists must be the same.")
    indices = [taxa.index(x) for x in nodes]

    return indices

def phylogenetic_tree(taxonomies):

    relationships = defaultdict(lambda : [])
    unaccounted_for = []
    for taxa in taxonomies:
        components = taxa.split('|')
        child = components[-1]
        if len(components) > 1:
            parent = components[-2]
            relationships[parent].append(child)
        else:
            unaccounted_for.append(child)

    G = nx.DiGraph()
    for child in unaccounted_for:
        G.add_node(child)
    for parent, children in relationships.items():
        for child in children:
            G.add_edge(parent, child)

    return G

def map_values_to_graph(graph, values_dict):

    values = [values_dict.get(node, 0) for node in graph.nodes()]

    return values 

def plot_filtered(df, threshold, multiply=400, ax=None):
    """ Extracts rows where abs > threshold and plots tree along with in between vertices """

    filtered = df.loc[abs(df[1])>threshold]
    
    cleaned_index = set()
    for x in filtered.index:
        components= x.split('|')
        for i in range(len(components)): 
            cleaned_index.add('|'.join(components[:i+1])) 
    cleaned = df.loc[cleaned_index]
    cleaned = cleaned*multiply
    plot_tree(cleaned, ax=ax)

def plot_tree(df, ax=None):

    graph = phylogenetic_tree(df.index)
    values_dict = {k:v[0] for k,v in zip(df.index, df.values)}
    mapped_values = map_values_to_graph(graph, values_dict)
    pos = graphviz_layout(graph, prog='twopi')
    nx.draw(graph, pos, cmap=plt.get_cmap('viridis'), node_color=mapped_values, with_labels=True, font_color='black', arrows=True, ax=ax)
    

def plot_intersection(df_dict, threshold, ax=None):
    """
    Given a dict of {label:df} containing attribution values, finds the index containing rows that are above threshold for every
    label and plots that tree.
    """
    thresholded = {k: df.loc[abs(df[1])>threshold]  for k,df in df_dict.items()}
    intersection = None
    for k, v in thresholded.items(): 
        if intersection is None: 
            intersection = v.index  
        else: 
            intersection = intersection.intersection(v.index)
    df = pd.DataFrame(index=intersection)
    df[1] = 1 # Set values for colormap
    plot_tree(df, ax=ax)
    return df

def plot_unique(df_dict, threshold, key, ax=None):
    """
    Given a dict of {label:df} containing attribution values, finds the index containing rows that are above threshold and
    only in key.
    """
    thresholded = {k: df.loc[abs(df[1])>threshold]  for k,df in df_dict.items()}
    index = thresholded[key].index
    for k, v in thresholded.items(): 
        if k == key: 
            pass
        else: 
            index = index.difference(v.index)
    df = pd.DataFrame(index=index)
    df[1] = 1 # Set values for colormap
    plot_tree(df, ax=ax)
    return df

def intersection_axes_generator(df_dict, axes, title="Attribution Tree at Threshold: {:.2f}"):

    ax = axes    
    def generator(frame, *fargs):
        ax.clear()
        e_frame = 10*np.exp(-.05*frame)
        ax.set_title(title.format(e_frame))
        plot_intersection(df_dict, e_frame, ax=ax)
        
    return generator

def threshold_generator(df, axes, disease, title="Attribution Tree at Threshold: {:.2f}"):

    ax = axes    
    def generator(frame, *fargs):
        ax.clear()
        e_frame = 10*np.exp(-.05*frame)
        ax.set_title(title.format(e_frame))
        plot_filtered(df, e_frame, ax=ax)

    return generator

def load_attributions(experiment_folder):
    files = {
        x: os.path.join(experiment_folder, x)
        for x in os.listdir(experiment_folder)
        if x.startswith('summed_')
    }
    files = {
        re.search('summed_([\w\W\d]+)\.csv',k).groups()[0]: v
        for k,v in files.items()
    }
    dfs = {
        k: pd.read_csv(v, index_col=0, header=None)
        for k,v in files.items()
    }
    return dfs

if __name__=="__main__":

    from matplotlib.animation import FuncAnimation
    import matplotlib
    import matplotlib.pyplot as plt

    dfs = load_attributions('./data/external/multiclass_25')
    fig, ax = plt.subplots()
    fig.set_size_inches(30,30)
    generator = intersection_axes_generator(dfs, ax)
    thresholds = [i for i in range(100)]
    animation = FuncAnimation(fig, generator, frames=thresholds)
    #with open("core.html",'w') as html:
        #video = animation.to_html5_video()
        #html.write(video)            
    """ from matplotlib.widgets import Slider
    axslider = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider = Slider(axslider, 'Log thresholds', 0, 200, valinit=0, valstep=1)
    slider.on_changed(generator)
    """
    matplotlib.rcParams['animation.embed_limit'] = 200
    for disease, df in dfs.items():
        fig, ax = plt.subplots()
        fig.set_size_inches(30,30)
        generator = threshold_generator(df, ax, disease)
        thresholds = [i for i in range(200)]
        animation = FuncAnimation(fig, generator, frames=thresholds)
        with open("{0}_thresholds.html".format(disease),"w") as html:
            print(disease)
            video = animation.to_html5_video()
            html.write(video)        