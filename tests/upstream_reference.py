"""Upstream DemoSpeedup, copied verbatim, so this project's ports have a reference.

Every function below is a byte-exact copy from ``lingxiao-guo/DemoSpeedup`` at
commit ``34bd43a832e19ac7aaa64acb5f19dce983075181``:

  * ``remove_outliers_isolation_forest``  robobase/robobase/utils.py
  * ``hdbscan_with_custom_merge``         robobase/robobase/utils.py

Copied rather than depended on. DemoSpeedup is a research repo, not a
distribution: its root has no ``pyproject.toml`` or ``setup.py``, ``robobase``
resolves through an SSH-only Gymnasium fork whose own manifest will not parse and
pins ``numpy<2`` against this project's numpy 2.x, and ``aloha``'s ``setup.py``
reads ``pkg_resources`` at build time without declaring it. None of the three
installs. These two functions are the entire surface this project still checks
itself against, so they live here and nothing else in the repo knows that
upstream exists.

**Do not tidy this code.** It is the reference the ports are compared to, so its
value is being unchanged: upstream's spelling, its dead assignments, its
misplaced parenthesis in ``hdbscan_with_custom_merge``'s cluster verdict. Fixing
anything here would silently redefine what "matches upstream" means. The ports
live in ``robot_stack.methods.demospeedup``; deliberate divergences from this
file are recorded and tested there.

One deviation, forced by reproducibility: upstream constructs its
``IsolationForest`` without a ``random_state``, so its labels differ run to run
and could not be asserted against. The name is bound below to a seeded partial;
the function bodies are untouched.
"""

# F841: upstream assigns `cluster_points` inside `split_large_clusters` and never
# reads it. That dead assignment is part of what is being copied; silencing the
# lint is correct here and editing it out is not.
# ruff: noqa: F841

import functools
import os

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest as _IsolationForest

#: Seeded so a comparison can be an equality rather than a distribution test.
IsolationForest = functools.partial(_IsolationForest, random_state=0)


def remove_outliers_isolation_forest(data, contamination=0.1):
    model = IsolationForest(contamination=contamination)
    predictions = model.fit_predict(data.reshape(-1, 1))
    
    data = data.copy() 

    if predictions[0] == -1: 
        next_idx = 1
        while next_idx < len(data) and predictions[next_idx] == -1:
            next_idx += 1
        if next_idx < len(data):  
            data[0] = data[next_idx]

    
    if predictions[-1] == -1:  
        prev_idx = len(data) - 2
        while prev_idx >= 0 and predictions[prev_idx] == -1:
            prev_idx -= 1
        if prev_idx >= 0:  
            data[-1] = data[prev_idx]

    
    for i in range(1, len(data) - 1):
        if predictions[i] == -1:  
            prev_idx = i - 1
            while prev_idx >= 0 and predictions[prev_idx] == -1:
                prev_idx -= 1 

            next_idx = i + 1
            while next_idx < len(data) and predictions[next_idx] == -1:
                next_idx += 1  

       
            if prev_idx >= 0 and next_idx < len(data):
                data[i] = (data[prev_idx] + data[next_idx]) / 2
            elif prev_idx >= 0:
                data[i] = data[prev_idx]
            elif next_idx < len(data):
                data[i] = data[next_idx]
                
    return data


def hdbscan_with_custom_merge(entropy, dir, rollout_id, plot=True):
    
    entropy = np.array(entropy)
    entropy_norm = (entropy-np.mean(entropy))/np.std(entropy)
    entropy_norm = remove_outliers_isolation_forest(entropy_norm)
    entropy_norm = (entropy_norm-np.mean(entropy_norm))/np.std(entropy_norm)
    indices = np.arange(len(entropy_norm))
    indices = (indices-np.mean(indices))/np.std(indices)
    X = np.stack((indices,entropy_norm),axis=-1)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    clusterer.fit(X)

    initial_labels = clusterer.labels_

    def split_large_clusters(labels, data, max_size=25):
        unique_labels = np.unique(labels)
        new_label = max(labels) + 1  

        for label in unique_labels:
            if label == -1: 
                continue

            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > max_size:
                cluster_points = data[cluster_indices]
                
                num_splits = len(cluster_indices) // max_size + (len(cluster_indices) % max_size > 0)
                
                for i in range(num_splits):
                    split_indices = cluster_indices[i * max_size:(i + 1) * max_size]
                    labels[split_indices] = new_label
                    new_label += 1  

        return labels
    
    initial_labels = split_large_clusters(initial_labels, X)
    
    unique_labels = np.unique(initial_labels[initial_labels >= 0])  
    
    refined_labels = np.full_like(initial_labels, -1)  

    for label in unique_labels:
        cluster_points = X[initial_labels == label]

        if  np.mean(cluster_points[:, 1] < 1):
            refined_labels[initial_labels == label] = 0  
        else:
            refined_labels[initial_labels == label] = -1  

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(entropy_norm)), entropy_norm, marker='o', markersize=5)  
        plt.title('1D Data Plot')
        plt.xlabel('Timestep')
        plt.ylabel('Entropy')
        plt.grid(True) 
        os.makedirs(os.path.join(dir, "plot"), exist_ok=True)
        plt.savefig(os.path.join(dir, f"plot/rollout{rollout_id}-entropy-curve.png"))
        plt.close()

    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=initial_labels, cmap='viridis', marker='o')
        plt.title('HDBSCAN Initial Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Cluster Label')
        os.makedirs(os.path.join(dir, "plot"), exist_ok=True)
        plt.savefig(os.path.join(dir, f"plot/rollout{rollout_id}-hdbscan-raw.png"))
        plt.close()

    if plot:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=refined_labels, cmap='viridis', marker='o')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Refined Cluster Label', rotation=270, labelpad=15)
        plt.title('HDBSCAN + Custom Merge Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.savefig(os.path.join(dir, f"plot/rollout{rollout_id}-hdbscan-refine.png"))
        plt.close()
    return np.abs(refined_labels)
