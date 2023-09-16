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
    eps = 0.5
    minpt = 5

    # Cluster data.
    cluster, label = dbscan(data, eps, minpt)

    print(f'There are {int(max(cluster))} clusters.')
    print(f'There are {np.sum(label==0)} unvisited points.')
    print(f'There are {np.sum(label==1)} core points.')
    print(f'There are {np.sum(label==2)} border points.')
    print(f'There are {np.sum(label==3)} noise points.')

    # Colours.
    colour = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Plot results.
    fig, ax = plt.subplots()
    for c in range(min(int(max(cluster)), len(colour))):
        # Plot core points.
        ax.scatter(data[(cluster==(c+1))&(label==1),0],
                   data[(cluster==(c+1))&(label==1),1], c=colour[c], marker='.')
        # Plot border points.
        ax.scatter(data[(cluster==(c+1))&(label==2),0],
                   data[(cluster==(c+1))&(label==2),1], marker='o', edgecolors=colour[c], facecolors='none')
    # Plot noise points.
    ax.scatter(data[label==3,0], data[label==3,1], c='k', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('DBSCAN')
    ax.set_aspect('equal')
    plt.show()

    return

def dbscan(data, eps, minpt):
    """
    data: numpy array with shape (n,m) where n is the number of points and m is the number of dimensions.
    eps: radius of neighbourhood with respect to some point.
    minpt: minimum number of points to form dense region.
    """

    # Number of points.
    n = data.shape[0]

    # Initialise cluster vector to store cluster index of point.
    cluster = np.zeros(n)

    # Initialise label vector to store point type (0:unvisited,1:core,2:border,3:noise).
    label = np.zeros(n)

    # Initialise current cluster index.
    cidx = 0

    for i in range(n):

        # Skip current point if visited.
        if label[i] != 0:
            continue

        # Calculate distance of other points from this point.
        dist = np.linalg.norm(data-data[i,:], axis=1)

        # Find all neighbours of current point (include current point).
        nb = np.where(dist < eps)[0]

        # Check if number of neighbours is sufficient.
        if nb.shape[0] < minpt:
            # Label current point as noise point.
            cluster[i] = 0
            label[i] = 3
            continue

        # Increment current cluster index.
        cidx += 1

        # Assign cluster index and label current point as core point.
        cluster[i] = cidx
        label[i] = 1

        # Initialise neighbour counter.
        j = 0

        # Iterate neighbours with counter.
        # While loop must be used as the number of neighbours is not constant.
        while j+1 <= nb.shape[0]:

            # Point index of this neighbour.
            k = nb[j]

            # Relabel noise point as border point and assign cluster index.
            if label[k] == 3:
                cluster[k] = cidx
                label[k] = 2

            # Skip this neighbour if visited (include relabeled noise point).
            if label[k] != 0:
                j += 1
                continue

            # Label this neighbour as border point and assign cluster index if unvisited.
            cluster[k] = cidx
            label[k] = 2

            # Calculate distance of other points from this neighbour.
            dist_nb = np.linalg.norm(data-data[k,:], axis=1)

            # Find all neighbours of this neighbour (include this neighbour).
            nb_ = np.where(dist_nb < eps)[0]

            # Check if number of neighbours of this neighbour is sufficient.
            if nb_.shape[0] >= minpt:
                # Label this neighbour as core point.
                label[k] = 1
                # Add all neighbours of this neighbour.
                tmp = np.setdiff1d(nb_, nb)
                nb = np.concatenate((nb, tmp))

            # Increment neighbour counter.
            j += 1

    return cluster, label

main()
