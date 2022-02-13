# -*- coding: utf-8 -*-
#  File: utilities.py
#  Project: 'OTUS.ML.ADV.HW3'
#  Created by Gennady Matveev (gm@og.ly) on 12-02-2022.
#  Copyright 2022. All rights reserved.

# Utility functions for otus_adv_hw3.ipynb

# Utility functions for cryptocompare downloads

# Import libraries

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist

from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import tsfel

import cryptocompare as cc

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly
from umap import UMAP

from kneed import KneeLocator
from tqdm.autonotebook import tqdm

import gdown
import shutil
import pickle
import requests

random_state = 17
scaler = StandardScaler()
n_jobs = -1


def set_api_key():

    cc.cryptocompare._set_api_key_parameter(
        "5db769e8ae211fc8c106e10623db6384dc64db9c26b2a2df708f8d1b53f99f92"
    )


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

    url_arch = (
        "https://drive.google.com/uc?export=download&id=1XCOhxPfRDp6SxMyPwPO1nse3MI2vOFvP"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1700.107 Safari/537.36"
    }

    # Download, if necessary
    if not os.path.exists("../data/Archive.zip"):
        # Check if data archive is still in Google Drive
        if requests.head(url_arch, headers=headers).status_code in [200, 302]:
            gdown.download(url_arch, output="../data/Archive.zip")
            shutil.unpack_archive("../data/Archive.zip", "../data")
            with open("../data/data_day.pickle", "rb") as f:
                data_day = pickle.load(f)
            with open("../data/data_hour.pickle", "rb") as f:
                data_hour = pickle.load(f)
            with open("../data/data_minute.pickle", "rb") as f:
                data_minute = pickle.load(f)
        # If not, get data from cryptocompare.com
        else:
            data_day = get_all_cc("day", 30)
            data_hour = get_all_cc("hour", 72)
            data_minute = get_all_cc("minute", 60)
    # Extract data from existing pickles
    else:
        with open("../data/data_day.pickle", "rb") as f:
            data_day = pickle.load(f)
        with open("../data/data_hour.pickle", "rb") as f:
            data_hour = pickle.load(f)
        with open("../data/data_minute.pickle", "rb") as f:
            data_minute = pickle.load(f)

    return data_day, data_hour, data_minute

    # Check CC list dataframe
    if not os.path.exists("../data/cryptocurrencies.pickle"):
        cc_url = "https://drive.google.com/uc?export=download&id=1Q09m-PfvhiBZ75lwXaHtafsajmXPW3WD"
        gdown.download(cc_url, output="../data/cryptocurrencies.pickle")
    with open("../data/cryptocurrencies.pickle", "rb") as f:
        ccs = pickle.load(f)
    # tickers = ccs["ticker"].to_list()


def pickle_data():
    with open("../data/data_minute.pickle", "wb") as f:
        pickle.dump(data_minute, f)
    with open("../data/data_hour.pickle", "wb") as f:
        pickle.dump(data_hour, f)
    with open("../data/data_day.pickle", "wb") as f:
        pickle.dump(data_day, f)

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
    _, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    ax1.plot(range(2, k_max), inertia, marker="s")
    ax1.set_title(f"The Elbow Method using Inertia\nmetric: {metric}")
    ax1.set_xlabel("Number of clusters")
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
#     clusters_no = data.loc["cluster"].nunique()
    color = ["g", "r", "b", "purple", "darkorange", "lightblue", "lightgreen"]
    for c in range(n_clusters):
        cc = color[c]
        fig, ax = plt.subplots(2, 4, sharex='col', figsize=(15, 5))
        cluster_ticks = data.T[data.T.loc[:, "cluster"] == c].index
        for i, tick in enumerate(cluster_ticks[:8]):
            ax[i % 2, i//2].plot(data.iloc[:-1][tick],
                                 color=cc)  # , label=tick)
            ax[i % 2, i//2].set_title(tick)
        fig.suptitle(f"Cluster {c}\n" + title, y=1.02)
        fig.show()

# ************************ TO BE REVISED *****************************


def visualize(X, model, **p):
    mdl = model(**p)
    X_clust = mdl.fit(X)
    X_color = X_clust.labels_.astype(str)
    features = X.values
    n_features = features.shape[1]

    # UMAP

    umap_3d = UMAP(n_components=3, init='random',
                   random_state=random_state)

    proj_3d = umap_3d.fit_transform(features)

    fig_3d = px.scatter_3d(
        proj_3d, x=0, y=1, z=2,
        color=X_color, labels={'color': 'clusters'},
        title=f"UMAP-3d projection",
        width=720, height=480,
        template="plotly_dark"
    )
    fig_3d.update_traces(marker_size=5)
    fig_3d.show()

    cln, cl_size = np.unique(X_color, return_counts=True)
    clusters = pd.DataFrame(cl_size, index=cln).T.rename(
        index=({0: "Cluster size"}))
    clusters.rename_axis(columns=["Cluster #"], inplace=True)
    return(clusters)

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

# ************************ TO BE REVISED *****************************

# Davies Bouldin score for K means


def get_kmeans_score(data, center):
    # instantiate kmeans
    kmeans = KMeans(n_clusters=center)
    # Then fit the model to your data using the fit method
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
    _, _ = plt.subplots(figsize=(15, 5))
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
    #     clusters_no = data.loc["cluster"].nunique()
    color = ["g", "r", "b", "purple", "darkorange", "lightblue", "lightgreen"]
    for c in range(n_clusters):
        cc = color[c]
        fig, ax = plt.subplots(2, 4, sharex='col', figsize=(15, 5))
        cluster_ticks = data.T[data.T.loc[:, "cluster"] == c].index
        for i, tick in enumerate(cluster_ticks[:8]):
            ax[i % 2, i//2].plot(data.iloc[:-1][tick],
                                 color=cc)  # , label=tick)
            ax[i % 2, i//2].set_title(tick)
        fig.suptitle(f"Cluster {c}\n" + title, y=1.02)
        fig.show()