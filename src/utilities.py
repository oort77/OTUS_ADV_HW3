# -*- coding: utf-8 -*-
#  File: utilities.py
#  Project: 'OTUS.ML.ADV.HW3'
#  Created by Gennady Matveev (gm@og.ly) on 12-02-2022.
#  Copyright 2022. All rights reserved.

# Utility functions for otus_adv_hw3.ipynb

# Import libraries

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist

from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import tsfel

import cryptocompare as cc

import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px
# import plotly
# from umap import UMAP

from kneed import KneeLocator
from tqdm.autonotebook import tqdm

import gdown
import shutil
import pickle
import requests

random_state = 17
scaler = StandardScaler()
n_jobs = -1

# Utility functions for cryptocompare downloads


def set_api_key():

    cc.cryptocompare._set_api_key_parameter(
        "5db769e8ae211fc8c106e10623db6384dc64db9c26b2a2df708f8d1b53f99f92"
    )


def get_tickers_list():
    with open("../data/tickers.txt","r") as f:
        tickers = f.read().split(', ')
    return tickers


def get_price(ticker: str, time_interval: str, limit: int):
    if time_interval == "day":
        result = cc.get_historical_price_day(ticker, currency="USD",
                                             limit=limit)  # exchange="Kraken",
    elif time_interval == "hour":
        result = cc.get_historical_price_hour(ticker, currency="USD",
                                              limit=limit)
    elif time_interval == "minute":
        result = cc.get_historical_price_minute(ticker, currency="USD",
                                                limit=limit)
    else:
        print("Please check your time_interval input")
    return result


def get_all_cc(time_interval: str, limit: int):
    df = pd.DataFrame(index=range(limit))
    tickers = get_tickers_list()
    for tick in tickers:
        print(tick, end="\t")
        try:
            d = get_price(tick, time_interval, limit)
            one_cc = pd.DataFrame.from_dict(d)["close"]
            one_cc.rename(index=tick, inplace=True)
            df = pd.concat([df, one_cc], axis=1)
            print("OK")
        except:
            print(f"{tick} passed")
    return df


def get_data():
    if os.path.isdir("../data") == False:
        os.mkdir("../data")

    # Download, if necessary
    if not os.path.exists("../data/../data/data_day.pickle"):
     # If not, get data from cryptocompare.com
        data_day = get_all_cc("day", 30)
        data_hour = get_all_cc("hour", 72)
        data_minute = get_all_cc("minute", 60)
    else:
        data_day = pd.read_pickle("../data/data_day.pickle")
        data_hour = pd.read_pickle("../data/data_hour.pickle")
        data_minute = pd.read_pickle("../data/data_minute.pickle")

    return data_day, data_hour, data_minute


def pickle_data(data_day, data_hour, data_minute):

    data_day.to_pickle("../data/data_day.pickle", protocol=4)
    data_hour.to_pickle("../data/data_hour.pickle", protocol=4)
    data_minute.to_pickle("../data/data_minute.pickle", protocol=4)

# Utility functions for clustering study


def plot_all_cc(data, title):
    X = scaler.fit_transform(data)
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(X)
    plt.suptitle(f"All CC scaled - {title}")
    plt.show()


def elbow_study(data, k_max: int = 10, metric="euclidean",
                model=TimeSeriesKMeans):
    X = scaler.fit_transform(data)
    inertia = []
    silhouette = []
    for k in tqdm(range(2, k_max)):
        if model == TimeSeriesKMeans:
            clusterer = model(n_clusters=k, metric=metric,
                              n_jobs=n_jobs, max_iter=10,
                              random_state=random_state)
        elif model == KMeans:
            clusterer = model(n_clusters=k, random_state=random_state)
        X_km = clusterer.fit(X)
        inertia.append(np.sqrt(X_km.inertia_))
        silhouette.append(silhouette_score(X, clusterer.labels_,
                                           metric=metric,
                                           random_state=random_state))
    # Find a knee
    kneedle = KneeLocator(range(2, k_max), inertia, S=2,
                          curve="convex", direction="decreasing")
    # Use 3 clusters in case kneed doesn't find a knee
    n_clusters = kneedle.knee or 3
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    ax1.plot(range(2, k_max), inertia, color='b', marker="s")
    ax1.set_title(f"The Elbow Method using Inertia\nmetric: {metric}")
    ax1.set_xlabel("Number of clusters", color='b')
    ax1.set_ylabel("Inertia")
    ax2.plot(range(2, k_max), silhouette, 'r-')
    ax2.set_ylabel('Silhouette', color='r')
    plt.plot(n_clusters, 0, color="r", marker="v")
    plt.show()

    print(f"Optimal number of clusters: {kneedle.knee}")

    return n_clusters


def plot_centroids(data, n_clusters, title, metric="euclidean"):
    X = scaler.fit_transform(data)
    ts_kmeans = TimeSeriesKMeans(n_clusters=n_clusters,
                                 metric=metric, n_jobs=n_jobs, max_iter=100)
    ts_kmeans.fit(X.T)
    fig, _ = plt.subplots(figsize=(15, 5))
    for cluster_number in range(n_clusters):
        plt.plot(ts_kmeans.cluster_centers_[cluster_number, :, 0].T,
                 label=cluster_number)
    plt.title(f"Cluster centroids\n{title}")
    plt.legend()
    plt.show()


def plot_clusters(data, n_clusters, title, metric="euclidean", model=TimeSeriesKMeans):

    if model == TimeSeriesKMeans:
        clusterer = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric,
                                     n_jobs=n_jobs, max_iter=100, random_state=random_state)
    else:
        clusterer = KMeans(n_clusters=n_clusters, max_iter=100,
                           random_state=random_state)
    X = scaler.fit_transform(data)
    data.loc["cluster"] = clusterer.fit_predict(X.T)
    clusters_no = data.loc["cluster"].value_counts(sort=False)
    color = ["g", "r", "b", "purple", "darkorange", "lightblue", "lightgreen"]
    for c in range(n_clusters):
        cc = color[c]
        fig, ax = plt.subplots(2, 4, sharex='col', figsize=(15, 5))
        cluster_ticks = data.T[data.T.loc[:, "cluster"] == c].index
        for i, tick in enumerate(cluster_ticks[:8]):
            ax[i % 2, i//2].plot(data.iloc[:-1][tick],
                                 color=cc)  # , label=tick)
            ax[i % 2, i//2].set_title(tick)
        fig.suptitle(f"Cluster {c}, {clusters_no[c]} items\n" + title, y=1.02)
        fig.show()

# ************************ TO BE REVISED *****************************


# def visualize(Xt, n_clusters):
#     clusterer = KMeans(n_clusters=n_clusters, max_iter=100,
#                        random_state=random_state)
#     # X_clust = clusterer.fit(Xt) #_predict
#     # X_color = X_clust.labels_.astype(str)
#     X = scaler.fit_transform(Xt.T)
#     X_clust = clusterer.fit_predict(X.T)
#     X_color = X_clust.T.astype(str)

#     features = Xt.T.values
#     # n_features = features.shape[1]

#     # UMAP
#     umap_3d = UMAP(n_components=3, init='random',
#                    random_state=random_state)

#     proj_3d = umap_3d.fit_transform(features)

#     fig_3d = px.scatter_3d(
#         proj_3d, x=0, y=1, z=2,
#         color=X_color, labels={'color': 'clusters'},
#         title=f"UMAP-3d projection",
#         width=720, height=480,
#         template="plotly_dark"
#     )
#     fig_3d.update_traces(marker_size=5)
#     fig_3d.show()

#     cln, cl_size = np.unique(X_color, return_counts=True)
#     clusters = pd.DataFrame(cl_size, index=cln).T.rename(
#         index=({0: "Cluster size"}))
#     clusters.rename_axis(columns=["Cluster #"], inplace=True)
#     return(clusters)

# ************************ TO BE REVISED *****************************


def clustering_study(data, metric, title):
    # Plot all scaled CC
    plot_all_cc(data, title)
    # Run elbow study
    n_clusters = elbow_study(data, metric=metric)
    # Plot centroids
    plot_centroids(data, n_clusters, title, metric=metric)
    # Show clusters composition (UMAP?)
    pass
    # Plot CC in clusters
    plot_clusters(data, n_clusters, title, metric=metric)

# Davies Bouldin score for K means


def get_kmeans_score(data, center):
    # Instantiate kmeans
    kmeans = KMeans(n_clusters=center)
    # Fit the model to data using the fit method
    model = kmeans.fit_predict(data)
    # Calculate Davies Bouldin score
    score = davies_bouldin_score(data, model)
    return score


def davies_bouldin_plot(data):
    scores = {}
    n_clusters = list(range(2, 10))
    for n in n_clusters:
        scores[n] = get_kmeans_score(data.T.values, n)
    n_cl = max(scores.keys(), key=(lambda k: scores[k]))

    _, _ = plt.subplots(figsize=(10, 5))
    plt.plot(n_clusters, scores.values(),
             linestyle='--', marker='s', color='b')
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies-Bouldin score")
    plt.title("Davies-Bouldin score vs. n_clusters")
    plt.plot(n_cl, 0, color="r", marker="v")
    plt.show()
    print(f"Optimal number of clusters: {n_cl}")
    return n_cl


def distances_distr_plot(data):
    _, ax = plt.subplots(1, 2, figsize=(15, 5))
    distr = pdist(data.T.values)
    plt.setp(ax[0], xlabel="Distance")
    plt.setp(ax[0], title="PDF")
    _ = sns.histplot(data=distr, bins=30, kde=True, ax=ax[0])

    plt.setp(ax[1], xlabel="Distance")
    plt.setp(ax[1], title="CDF")
    _ = sns.histplot(data=distr, bins=30, kde=True,
                     cumulative=True, color="g", ax=ax[1])


def plot_clusters_2(data, Xt, n_clusters, random_state, title):
    clusterer = KMeans(n_clusters=n_clusters, max_iter=100,
                       random_state=random_state)
    X = scaler.fit_transform(Xt)
    data.loc["cluster"] = clusterer.fit_predict(X.T)
    color = ["g", "r", "b", "purple", "darkorange", "lightblue", "lightgreen"]
    clusters_no = data.loc["cluster"].value_counts(sort=False)

    for c in range(n_clusters):
        cc = color[c]
        fig, ax = plt.subplots(2, 4, sharex='col', figsize=(15, 5))
        cluster_ticks = data.T[data.T.loc[:, "cluster"] == c].index
        for i, tick in enumerate(cluster_ticks[:8]):
            ax[i % 2, i//2].plot(data.iloc[:-1][tick],
                                 color=cc)  # , label=tick)
            ax[i % 2, i//2].set_title(tick)
        fig.suptitle(f"Cluster {c}, {clusters_no[c]} items\n" + title, y=1.02)
        fig.show()
    return(data.copy())
