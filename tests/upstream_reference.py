"""Upstream DemoSpeedup, copied verbatim, so this project's ports have a reference.

Every function below is a byte-exact copy from ``lingxiao-guo/DemoSpeedup`` at
commit ``34bd43a832e19ac7aaa64acb5f19dce983075181``:

  * ``remove_outliers_isolation_forest``  robobase/robobase/utils.py
  * ``hdbscan_with_custom_merge``         robobase/robobase/utils.py
  * ``process_action_label``              aloha/act/act_utils.py

Copied rather than depended on. DemoSpeedup is a research repo, not a
distribution: its root has no ``pyproject.toml`` or ``setup.py``, ``robobase``
resolves through an SSH-only Gymnasium fork whose own manifest will not parse and
pins ``numpy<2`` against this project's numpy 2.x, and ``aloha``'s ``setup.py``
reads ``pkg_resources`` at build time without declaring it. None of the three
installs. These three functions are the entire surface this project checks itself
against, so they live here and nothing else in the repo knows that upstream
exists.

**Do not tidy this code.** It is the reference the ports are compared to, so its
value is being unchanged: upstream's spelling, its dead assignments, its
misplaced parenthesis in ``hdbscan_with_custom_merge``'s cluster verdict. Fixing
anything here would silently redefine what "matches upstream" means. The ports
live in ``pace_bench.methods.demospeedup``; deliberate divergences from this
file are recorded and tested there.

One deviation, forced by reproducibility: upstream constructs its
``IsolationForest`` without a ``random_state``, so its labels differ run to run
and could not be asserted against. The name is bound below to a seeded partial;
the function bodies are untouched.
"""

# Upstream leaves two values unused: `cluster_points` inside `split_large_clusters`
# (F841) and `dim` unpacked from `action.shape` (RUF059). Both are part of what is
# being copied; silencing the lint is correct here and editing them out is not.
# ruff: noqa: F841, RUF059

import functools
import os

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import torch
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


def process_action_label(action, label, is_pad):
    low_v = 2
    high_v = 4
    horizon, dim = action.shape
    new_actions = torch.zeros_like(action)
    new_labels = torch.zeros_like(label)
    new_is_pad = torch.zeros_like(is_pad)

    current_action = action  # Shape: (horizon, dim)
    current_label = label  # Shape: (horizon,)
    current_is_pad = is_pad

    indices = []
    i = -1
    while i < horizon:
        if current_label[i] == 0 and i+low_v < horizon:
            i += low_v  # Skip next element
            indices.append(i)
        elif current_label[i] == 1:
            # Check the next high_v elements if they exist
            if i + high_v < horizon and torch.all(current_label[i:i + high_v] == 1):
                i += high_v  # Skip the next 3 elements
                indices.append(i)
            else:
                # Find the next 0 element if it exists
                next_zero = (current_label[i + 1:] == 0).nonzero(as_tuple=True)[0]
                if len(next_zero) > 0:
                    i = i + 1 + next_zero[0].item()
                    indices.append(i)
                else:
                    break  # No more 0s, stop
        else:
            i += 1
    
    # Use the indices to extract new action and label
    new_actions[:len(indices)] = current_action[indices]
    new_labels[:len(indices)] = current_label[indices]
    new_is_pad[:len(indices)] = current_is_pad[indices]

    return new_actions, new_is_pad
