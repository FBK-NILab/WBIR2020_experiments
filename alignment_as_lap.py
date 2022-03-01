"""Alignment of tractograms as LAP.

Copyright Emanuele Olivetti, 2018, 2020, 2022
BSD License, 3 clauses.

"""

import numpy as np
from nibabel.streamlines import load
from dissimilarity import compute_dissimilarity, dissimilarity
from kmeans import mini_batch_kmeans, compute_labels, compute_centroids
from scipy.spatial import KDTree
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.distances import bundles_distances_mam, bundles_distances_mdf
from dipy.tracking.streamlinespeed import length
from scipy.optimize import linear_sum_assignment
# from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import min_weight_full_bipartite_matching  # this is the sparse equivalent to linear_sum_assignment
from scipy.sparse import csc_matrix


try:
    from joblib import Parallel, delayed
    joblib_available = True
except ImportError:
    joblib_available = False


def load_tractogram(T_filename, threshold_short_streamlines=10.0, nb_points=16):
    """Load tractogram from TRK file, resample to the desired number of
    points and remove short streamlines with length below threshold.
    """
    print("Loading %s" % T_filename)
    data = load(T_filename)
    T = data.streamlines
    print("%s: %s streamlines" % (T_filename, len(T)))

    if nb_points is not None:
        print("Resampling all streamlines to %s points" % (nb_points,))
        T = set_number_of_points(T, nb_points)

    T = np.array(T, dtype=object)
    
    if threshold_short_streamlines > 0:
        # Removing short artifactual streamlines
        print("Removing (presumably artifactual) streamlines shorter than %s" % threshold_short_streamlines)
        T = np.array([s for s in T if length(s) >= threshold_short_streamlines], dtype=object)
        print("%s: %s streamlines" % (T_filename, len(T)))

    return T


def clustering(S_dr, k, b=100, t=100, S_centers=None):
    """Wrapper of the mini-batch k-means algorithm that combines multiple
    basic functions into a convenient one.
    """
    if S_centers is None:
        # Generate k random centers:
        S_centers = S_dr[np.random.permutation(S_dr.shape[0])[:k]]
        # Improve the k centers with mini_batch_kmeans
        S_centers = mini_batch_kmeans(S_dr, S_centers, b=b, t=t)
    else:
        # First implementation:
        #
        # print("Findind the new centers as the closest to the given ones")
        # kdt = KDTree(S_dr)
        # _, S_centers_idx = kdt.query(S_centers, k=1, workers=-1)
        # assert(np.unique(S_centers_idx).size == S_centers_idx.size) # assert that there are no duplicate centers ## WARNING!!! THIS SOMETIMES FAILS!!!
        # S_centers = S_dr[S_centers_idx]
        #
        # Second implementation:
        #
        # First attempt of the second implementation:
        # print("Using LAP to find the new new centers as the closest to the given ones")
        # dm_S_centers_S_dr = cdist(S_centers, S_dr) # THIS REQUIRES TOO MUCH MEMORY!!
        # _, S_centers_idx = linear_sum_assignment(dm_S_centers_S_dr)
        # S_centers = S_dr[S_centers_idx]
        #
        # Second attempt of the second implementation:
        print("Using sparse LAP to find the new new centers as the closest to the given ones")
        print("Computing the closest candidates to S_centers, for the sparse LAP")
        kdt = KDTree(S_dr)
        k = 50
        data, col = kdt.query(S_centers, k=k, workers=-1)  # what is a value for k which is usually good?
        data = data + 1.0e-10  # this prevents problems when there are perfect matches with zero distance that are considered missing values by the sparse LAP!
        print("Creating a sparse distance matrix with the result")
        row = np.repeat(np.arange(len(S_centers), dtype=int), k)
        dm_S_centers_S_dr_sparse = csc_matrix((data.flatten(), (row, col.flatten())), shape=(len(S_centers), len(S_dr)))
        print("Solving the sparse LAP")
        _, S_centers_idx = min_weight_full_bipartite_matching(dm_S_centers_S_dr_sparse)
        S_centers = S_dr[S_centers_idx]

    # Assign the cluster labels to each streamline. The label is the
    # index of the nearest center.
    S_cluster_labels = compute_labels(S_dr, S_centers)

    # Compute a cluster representive, for each cluster
    S_representatives_idx = compute_centroids(S_dr, S_centers)
    return S_representatives_idx, S_cluster_labels


def LAP(S_A, S_B, verbose=True, parallel=True, distance_function=bundles_distances_mam):
    """
    """
    assert(len(S_B) >= len(S_A))  # required by LAP
    if verbose:
        print("Computing LAP between streamlines.")
        print("Computing the distance matrix between the two sets of streamlines.")

    dm_AB = streamline_distance(S_A, S_B, parallel=parallel,
                                distance_function=distance_function)
    _, corresponding_streamlines = linear_sum_assignment(dm_AB)
    return corresponding_streamlines


def streamline_distance(S_A, S_B=None, parallel=True,
                        distance_function=bundles_distances_mam):
    """Wrapper to decide what streamline distance function to use. The
    function computes the distance matrix between sets of
    streamlines. This implementation provides optimiztions like
    parallelization and avoiding useless computations when S_B is
    None.
    """
    # distance_function = bundles_distances_mdf
    if parallel:
        return dissimilarity(S_A, S_B, distance_function)
    else:
        return distance_function(S_A, S_B)


def distance_corresponding(A, B, correspondence,
                           distance_function=bundles_distances_mam):
    """Distance between streamlines in set A and the corresponding ones in
    B. The vector 'correspondence' has in position 'i' the index of
    the streamline in B that corresponds to A[i].

    """
    return np.array([streamline_distance([A[i]],
                                         [B[correspondence[i]]],
                                         parallel=False,
                                         distance_function=distance_function)
                     for i in range(len(A))]).squeeze()


def LAP_two_clusters(cluster_A, cluster_B, alpha=0.5, max_iter1=100,
                     max_iter2=100, parallel=True,
                     distance_function=bundles_distances_mam):
    """Wrapper of LAP() between the streamlines of two clusters. This code
    is able two handle clusters of different sizes and to invert the
    result of corresponding_streamlines, if necessary.
    """
    if len(cluster_A) <= len(cluster_B):  # LAP(A,B)
        corresponding_streamlines = LAP(cluster_A, cluster_B,
                                        verbose=False,
                                        parallel=parallel,
                                        distance_function=distance_function)
    else:  # LAP(B,A)
        corresponding_streamlines = LAP(cluster_B, cluster_A,
                                        verbose=False,
                                        parallel=parallel,
                                        distance_function=distance_function)
        # invert result from B->A to A->B:
        tmp = -np.ones(len(cluster_A), dtype=int)
        for j, v in enumerate(corresponding_streamlines):
            if v != -1:
                tmp[v] = j

        corresponding_streamlines = tmp

    return corresponding_streamlines


def LAP_all_corresponding_pairs(T_A, T_B, k,
                                T_A_cluster_labels,
                                T_B_cluster_labels,
                                corresponding_clusters,
                                distance_function=bundles_distances_mam):
    """Loop over all pairs of correponding clusters and perform graph
    matching between the streamlines of correponding clusters.

    This code executes parallel (multicore) for-loop, if joblib is
    available. If not, it reverts to a standard for-loop.
    """
    print("Compute LAP between streamlines of corresponding clusters")
    correspondence_lap = -np.ones(len(T_A), dtype=int)  # container of the results
    if joblib_available:
        print("Parallel version: executing %s tasks in parallel" % k)
        n_jobs = -1
        clusters_A_idx = [np.where(T_A_cluster_labels == i)[0] for i in range(k)]
        clusters_A = [T_A[clA_idx] for clA_idx in clusters_A_idx]
        clusters_B_idx = [np.where(T_B_cluster_labels == corresponding_clusters[i])[0] for i in range(k)]
        clusters_B = [T_B[clB_idx] for clB_idx in clusters_B_idx]
        results = Parallel(n_jobs=n_jobs, verbose=True)(delayed(LAP_two_clusters)(clusters_A[i], clusters_B[i], parallel=False, distance_function=distance_function) for i in range(k))
        # merge results
        for i in range(k):
            tmp = results[i] != -1
            correspondence_lap[clusters_A_idx[i][tmp]] = clusters_B_idx[i][results[i][tmp]]

    else:
        for i in range(k):
            print("LAP between streamlines of corresponding clusters: cl_A=%s <-> cl_B=%s" % (i, corresponding_clusters[i]))
            cluster_A_idx = np.where(T_A_cluster_labels == i)[0]
            cluster_A = T_A[cluster_A_idx]
            cluster_B_idx = np.where(T_B_cluster_labels == corresponding_clusters[i])[0]
            cluster_B = T_B[cluster_B_idx]
            corresponding_streamlines = LAP_two_clusters(cluster_A,
                                                         cluster_B,
                                                         distance_function=distance_function)

            tmp = corresponding_streamlines != -1
            correspondence_lap[cluster_A_idx[tmp]] = cluster_B_idx[corresponding_streamlines[tmp]]

    return correspondence_lap


def fill_missing_correspondences(correspondence_lap, T_A_dr):
    """After LAP, in case some correspondences are missing,
    i.e. their target index is '-1', fill them following this idea:
    for a given streamline T_A[i], its correponding one in T_B is the
    one corresponding to the nearest neighbour of T_A[i] in T_A.

    The (approximate nearest neighbour) is computed with a KDTree on
    the dissimilarity representation of T_A, i.e. T_A_dr.
    """
    print("Filling missing correspondences in T_A with the corresponding to their nearest neighbour in T_A")
    correspondence = correspondence_lap.copy()
    T_A_corresponding_idx = np.where(correspondence != -1)[0]
    T_A_missing_idx = np.where(correspondence == -1)[0]
    T_A_corresponding_kdt = KDTree(T_A_dr[T_A_corresponding_idx])
    # T_A_missing_NN = T_A_corresponding_kdt.query(T_A_dr[T_A_missing_idx],
    #                                              k=1,
    #                                              return_distance=False,
    #                                              workers=-1).squeeze()
    _, T_A_missing_NN = T_A_corresponding_kdt.query(T_A_dr[T_A_missing_idx],
                                                    k=1,
                                                    workers=-1)
    correspondence[T_A_missing_idx] = correspondence[T_A_corresponding_idx[T_A_missing_NN]]
    return correspondence


def alignment_as_LAP(T_A, T_B,
                     k, threshold_short_streamlines=10.0,
                     b=100,
                     t=100,
                     T_A_dr=None,  # precomputed dissimilarity representation, useful for large experiments
                     T_B_dr=None,
                     distance_function=bundles_distances_mam):

    # 2) Compute the dissimilarity representation of T_A and T_B
    if T_A_dr is None:
        print("Computing the dissimilarity representation of T_A")
        T_A_dr, prototypes_A = compute_dissimilarity(T_A, distance=distance_function)

    if T_B_dr is None:
        print("Computing the dissimilarity representation of T_B")
        # T_B_dr, prototypes_B = compute_dissimilarity(T_B, distance=distance_function)
        print("Using prototypes of T_A")
        T_B_dr = dissimilarity(T_B, T_A[prototypes_A], distance=distance_function)

    # 3) Compute the k-means clustering of T_A and T_B
    b = 100  # mini-batch size
    t = 100  # number of iterations
    print("Computing the k-means clustering of T_A and T_B, k=%s" % k)
    print("mini-batch k-means on T_A")
    T_A_representatives_idx, T_A_cluster_labels = clustering(T_A_dr, k=k, b=b, t=t)
    print("mini-batch k-means on T_B")
    print("...using T_A_representatives")
    T_B_representatives_idx, T_B_cluster_labels = clustering(T_B_dr, k=k, b=b, t=t, S_centers=T_A_dr[T_A_representatives_idx])
    tmp = np.unique(T_B_cluster_labels).size
    print("How many clustes were formed? %s" % (tmp,))
    assert(tmp == k) # check that all clusters were formed

    # 4) Compute LAP between T_A_representatives and T_B_representatives
    alpha = 0.5
    max_iter1 = 100
    max_iter2 = 100
    corresponding_clusters = LAP(T_A[T_A_representatives_idx],
                                 T_B[T_B_representatives_idx],
                                 distance_function=distance_function)
    distance_clusters = distance_corresponding(T_A[T_A_representatives_idx],
                                               T_B[T_B_representatives_idx],
                                               corresponding_clusters,
                                               distance_function=distance_function)
    print("Median distance between corresponding clusters: %s" % np.median(distance_clusters))

    # 5) For each pair corresponding cluster, compute LAP
    # between their streamlines
    correspondence_lap = LAP_all_corresponding_pairs(T_A, T_B, k,
                                                     T_A_cluster_labels,
                                                     T_B_cluster_labels,
                                                     corresponding_clusters,
                                                     distance_function=distance_function)

    # 6) Filling the missing correspondences in T_A with the
    # correspondences of the nearest neighbors in T_A
    correspondence = fill_missing_correspondences(correspondence_lap, T_A_dr)

    # 7) Compute the mean distance of corresponding streamlines, to
    # check the quality of the result
    distances = distance_corresponding(T_A, T_B, correspondence,
                                       distance_function=distance_function)
    print("Median distance of corresponding streamlines: %s" % np.median(distances))

    return correspondence, distances


if __name__ == '__main__':
    print(__doc__)
    np.random.seed(0)

    T_A_filename = 'data/sub-500222/dt-neuro-track-trk.tag-ensemble.tag-t1.id-605a1e1c73b69eaef86d2f54/track.trk'
    T_B_filename = 'data/sub-506234/dt-neuro-track-trk.tag-ensemble.tag-t1.id-605a1ee373b69e81bf6d34f9/track.trk'
    # T_B_filename = T_A_filename

    distance_function = bundles_distances_mam  # bundles_distances_mdf is faster
    
    # Main parameters:
    k = 5000  # number of clusters, usually somewhat above sqrt(|T_A|) is optimal for efficiency.
    threshold_short_streamlines = 0.0  # Beware: discarding streamlines affects IDs

    # Additional internal parameters for mini-batch k-means, no need to change them, usually:
    b = 100
    t = 100

    # Number of points to which resample each streamline, for performance reasons:
    nb_points = 16

    # 1) load T_A and T_B
    T_moving = load_tractogram(T_A_filename,
                               threshold_short_streamlines=threshold_short_streamlines,
                               nb_points=nb_points)
    T_static = load_tractogram(T_B_filename,
                               threshold_short_streamlines=threshold_short_streamlines,
                               nb_points=nb_points)

    correspondence, distances = alignment_as_LAP(T_A=T_moving,
                                                 T_B=T_static,
                                                 k=k,
                                                 threshold_short_streamlines=threshold_short_streamlines,
                                                 b=b,
                                                 t=t,
                                                 distance_function=distance_function)

    print("Saving the result into correspondence.csv")
    result = np.vstack([range(len(correspondence)), correspondence]).T
    np.savetxt("correspondence.csv", result, fmt='%d', delimiter=',',
               header='ID_A,ID_B')

    import matplotlib.pyplot as plt
    plt.interactive(True)
    plt.figure()
    plt.hist(distances, bins=50)
    plt.title("Distances between corresponding streamlines")
