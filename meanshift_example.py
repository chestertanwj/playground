import numpy as np
import matplotlib.pyplot as plt

def main():

    # Generate data.
    np.random.seed(0)
    data = np.random.uniform(low=-5, high=5, size=(200,2))
    cov = np.array([[0.1,0],[0,0.1]])
    for i in range(3):
        for j in range(3):
            data = np.concatenate((data, np.random.multivariate_normal(np.array([3-i*3,3-j*3]), cov, 50)))

    # Set parameters.
    bw = 2
    tol = 0.01
    n = 20

    # Cluster data.
    cluster, centroid = meanshift(data, bw, tol)

    print(f'There are {int(max(cluster)+1)} clusters.')

    # Colours.
    colour = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Plot results.
    fig, ax = plt.subplots()
    c = 0
    d = 0
    while True:
        if np.count_nonzero(cluster==c) >= n:
            # Plot points of clusters above a certain size.
            ax.scatter(data[(cluster==c),0], data[(cluster==c),1], c=colour[d], marker='.')
            # Plot centroids of cluster.
            ax.scatter(centroid[(cluster==c),0], centroid[(cluster==c),1],
                       marker='o', edgecolors='k', facecolors=colour[d])
            d += 1
        c += 1
        # Check if clusters or colours exhausted.
        if c > int(max(cluster)) or d > len(colour)-1:
            break
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('MEANSHIFT')
    ax.set_aspect('equal')
    plt.show()

    return

def meanshift(data, bw, tol):
    """
    data: numpy array with shape (n,m) where n is the number of points and m is the number of dimensions.
    bw: bandwidth to determine window size for mean computation.
    tol: tolerance to determine mean convergence.
    """

    # Number of points and number of dimensions.
    n_pt, n_dim = data.shape

    # Initialise centroid array to store centroid of cluster.
    centroid = np.zeros(data.shape)

    for i in range(n_pt):

        # Initialise current mean with current point.
        mean_curr = data[i,:]

        # Execute mean shift.
        while True:

            # Calculate distance of other points from current mean.
            dist = np.linalg.norm(data-mean_curr, axis=1)

            # Find all points in mean shift window.
            win = np.where(dist < bw)[0]

            # Calculate next mean.
            mean_next = np.mean(data[win, :], axis=0)

            # Calculate difference between current mean and next mean.
            diff = np.linalg.norm(mean_curr-mean_next)

            # Check for mean convergence.
            if diff < tol:
                break

            # Assign next mean to current mean.
            mean_curr = mean_next

        # Assign final mean to centroid vector.
        centroid[i,:] = mean_next

    # Sort rows of centroid array.
    sort_idx = np.lexsort(np.transpose(centroid))
    centroid_sort = centroid[sort_idx, :]

    # Length of tolerance vector in each dimension.
    tol_len = np.sqrt(tol/n_dim)
    # Get rounding decimals based on tolerance vector.
    tol_dec = np.max([0, np.ceil(-np.log10(tol_len))])

    # Round centroid array according to rounding decimals.
    centroid_round = np.round(centroid_sort, int(tol_dec))
    # Uniquify centroid array and get inverse indices.
    _, unq_inv = np.unique(centroid_round, return_inverse=True, axis=0)

    # Find cluster vector by unsorting inverse indices.
    # Cluster vector stores cluster index of points.
    cluster = unq_inv[np.argsort(sort_idx)]

    # Unsort rows of centroid array.
    centroid = centroid_sort[np.argsort(sort_idx), :]

    return cluster, centroid

main()
