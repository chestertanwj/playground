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
    ordering, reach_dist = optics(data, eps, minpt)
    reach_dist_order = reach_dist[ordering]

    # Colours.
    colour = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Plot reachability.
    fig1, ax1 = plt.subplots()
    plt.plot(reach_dist_order)
    ax1.set_xlabel('Ordering')
    ax1.set_ylabel('Reachability Distance')
    ax1.set_title('Reachability Plot')

    # Plot results.
    fig2, ax2 = plt.subplots()
    ax2.scatter(data[:,0], data[:,1], c='k', marker='.')
    c = 0
    idx1 = float('nan')
    idx2 = float('nan')
    for i in range(data.shape[0]):
        if c == len(colour):
            # No more colours.
            break
        if i == data.shape[0]-1:
            # End of data.
            if not np.isnan(reach_dist_order[i]):
                ax2.scatter(data[ordering[idx1:i],0], data[ordering[idx1:i],1], c=colour[c], marker='.')
            break
        if np.isnan(reach_dist_order[i]) and not np.isnan(reach_dist_order[i+1]):
            # Start of cluster.
            idx1 = i+1
        elif not np.isnan(reach_dist_order[i]) and np.isnan(reach_dist_order[i+1]):
            # End of cluster.
            idx2 = i+1
            ax2.scatter(data[ordering[idx1:idx2],0], data[ordering[idx1:idx2],1], c=colour[c], marker='.')
            c += 1
        else:
            pass
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('OPTICS')
    ax2.set_aspect('equal')
    plt.show()

    return

def optics(data, eps, minpt):
    """
    data: numpy array with shape (n,m) where n is the number of points and m is the number of dimensions.
    eps: radius of neighbourhood with respect to some point.
    minpt: minimum number of points to form dense region.
    """

    # Number of points.
    n = data.shape[0]

    # Initialise core distance vector.
    core_dist = np.zeros(n)
    core_dist.fill(np.nan)

    # Initialise reachability distance vector.
    reach_dist = np.zeros(n)
    reach_dist.fill(np.nan)

    # Initialise ordering.
    ordering = np.array([], dtype='int')

    # Calculate core distance of points.
    for i in range(n):
        # Calculate distance of other points from this point.
        dist = np.linalg.norm(data-data[i,:], axis=1)
        # Find all neighbours of current point (include current point).
        nb = np.where(dist < eps)[0]
        # Check if number of neighbours is sufficient.
        if nb.shape[0] >= minpt:
            # Sort neighbours according to their distances.
            dist_nb = dist[nb]
            dist_nb_sort = np.sort(dist_nb)
            # Assign minpt-th smallest distance as core distance.
            core_dist[i] = dist_nb_sort[minpt-1]

    # Initialise unprocessed points.
    unprocessed = np.arange(n)

    # Define update function.
    def update(nb, pt):
        nonlocal seed_p
        nonlocal seed_q
        nonlocal data
        nonlocal unprocessed
        nonlocal core_dist
        nonlocal reach_dist

        for k in nb:
            if k in unprocessed:
                # Calculate new reachability distance.
                new_reach_dist = max(core_dist[pt], np.linalg.norm(data[pt,:]-data[k,:]))
                if np.isnan(reach_dist[k]):
                    # Assign reachability distance.
                    reach_dist[k] = new_reach_dist
                    # Add point to priority queue.
                    seed_p = np.append(seed_p, new_reach_dist)
                    seed_q = np.append(seed_q, k)
                else:
                    if new_reach_dist < reach_dist[k]:
                        # Update reachability distance.
                        reach_dist[k] = new_reach_dist
                        # Update point in priority queue.
                        seed_p[np.argwhere(seed_q==k)] = new_reach_dist

    # Iterate unprocessed points.
    while unprocessed.size > 0:
        # Point index of unprocessed point.
        i = unprocessed[0]
        # Mark point as processed and assign to ordering.
        unprocessed = unprocessed[1:]
        ordering = np.append(ordering, i)

        # Calculate distance of other points from this point.
        dist = np.linalg.norm(data-data[i,:], axis=1)
        # Find all neighbours of current point (include current point).
        nb = np.where(dist < eps)[0]

        # Check if number of neighbours is sufficient.
        if nb.shape[0] >= minpt:
            # Initialise priority queue (distances and indices).
            seed_p = np.array([])
            seed_q = np.array([], dtype='int')
            # Update priority queue.
            update(nb, i)

            # Iterate points in priority queue.
            while seed_q.size > 0:
                # Find next point in priority queue.
                idx = np.argwhere(seed_p==np.min(seed_p))[0]
                # Point index of next point in priority queue.
                j = seed_q[idx]
                # Remove point from priority queue.
                seed_p = np.delete(seed_p, idx)
                seed_q = np.delete(seed_q, idx)
                # Mark point as processed and assign to ordered list.
                unprocessed = np.setdiff1d(unprocessed, j)
                ordering = np.append(ordering, j)

                # Calculate distance of other points from this point.
                dist = np.linalg.norm(data-data[j,:], axis=1)
                # Find all neighbours of current point (include current point).
                nb_ = np.where(dist < eps)[0]

                # Check if number of neighbours is sufficient.
                if nb_.shape[0] >= minpt:
                    # Update priority queue.
                    update(nb_, j)

    return ordering, reach_dist

main()
