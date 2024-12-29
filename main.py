import random
import numpy as np
import matplotlib.pyplot as plt

MIN_COORD = -5000
MAX_COORD = 5000
MAX_OFFSET = 100
NUMBER_START_POINTS = 20
NUMBER_START_CLUSTER = 10
MAX_AVERAGE_DISTANCE = 500
NUMBER_OF_POINTS = 40000
STEP_BY_STEP_VISUALIZATION = False
MODE = 1 # 1 - knn_clustering_centroid / 2 - knn_clustering_medoid / 3 - divisive_clustering_centroid



def generate_start_positions():
    x_coords = np.random.randint(MIN_COORD, MAX_COORD + 1, size=NUMBER_START_POINTS)
    y_coords = np.random.randint(MIN_COORD, MAX_COORD + 1, size=NUMBER_START_POINTS)
    start_positions = np.column_stack((x_coords, y_coords))
    return start_positions


def point_by_offset(point):
    max_x = min(MAX_COORD - point[0], MAX_OFFSET)
    min_x = max(MIN_COORD - point[0], -MAX_OFFSET)
    max_y = min(MAX_COORD - point[1], MAX_OFFSET)
    min_y = max(MIN_COORD - point[1], -MAX_OFFSET)

    x = random.randint(min_x, max_x)
    y = random.randint(min_y, max_y)

    new_x = point[0] + x
    new_y = point[1] + y

    return [new_x, new_y]


def generate_points():
    # Генерация стартовых позиций
    start_positions = generate_start_positions()

    # Предварительное выделение памяти для всех точек
    point_array = np.zeros((NUMBER_START_POINTS + NUMBER_OF_POINTS, 2), dtype=int)
    point_array[:NUMBER_START_POINTS] = start_positions

    # Генерация дополнительных точек с учетом смещений
    for i in range(NUMBER_OF_POINTS):
        index = random.randint(0, NUMBER_START_POINTS + i - 1)
        point = point_by_offset(point_array[index])
        point_array[NUMBER_START_POINTS + i] = point

    return point_array


def visualisation(clusters, centroids):
    num_clusters = len(clusters)
    colormap = plt.colormaps['viridis']

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        if cluster.size > 0:
            plt.scatter(cluster[:, 0], cluster[:, 1], color=colormap(i / (num_clusters - 1)), s=1)

    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='o', s=20)
    plt.title('K-Means Clustering')
    plt.show()


def knn_clustering_centroid(points):
    clusters = None
    centroids = None
    first_run = True
    need_continue = True
    max_distance_point = None
    iteration = 1

    while need_continue:
        if first_run:
            first_run = False
            random_indices = np.random.choice(points.shape[0], NUMBER_START_CLUSTER, replace=False)
            centroids = points[random_indices]
        else:
            # append to np array the point with max distance
            centroids = np.vstack([centroids, max_distance_point])

        clusters = [[] for _ in range(len(centroids))]

        # find for every point centroid with minimum distance and add to this cluster
        for point in points:
            distances = np.linalg.norm(centroids - point, axis=1) # np.linalg.norm - find the euclid value, as the result we have the 1d array with distance to every centroid
            nearest_centroid_idx = np.argmin(distances)
            clusters[nearest_centroid_idx].append(point)


        need_continue = False
        max_distance_point = None
        max_average_distance = -1


        for i in range(len(clusters)):
            if clusters[i]:
                new_centroid = np.mean(clusters[i], axis=0)
                distances = [np.linalg.norm(point - new_centroid) for point in clusters[i]]
                average_distance = np.mean(distances)

                if average_distance > MAX_AVERAGE_DISTANCE:
                    need_continue = True

                if average_distance > max_average_distance:
                    max_average_distance = average_distance
                    max_distance_point = clusters[i][np.argmax(distances)]

                    # random
                    # random_indices = np.random.choice(len(clusters[i]), 1, replace=False)
                    # max_distance_point = (clusters[i])[random_indices[0]]

                centroids[i] = new_centroid

        print(f"Iteration {iteration}: Max average distance = {max_average_distance}")
        iteration += 1
        if (len(clusters) > 1 and STEP_BY_STEP_VISUALIZATION):
            visualisation(clusters, centroids)

    return centroids, clusters


def knn_clustering_medoid(points):
    clusters = None
    medoids = None
    first_run = True
    need_continue = True
    max_distance_point = None
    iteration = 1

    while need_continue:
        if first_run:
            first_run = False
            random_indices = np.random.choice(points.shape[0], NUMBER_START_CLUSTER, replace=False)
            medoids = points[random_indices]
        else:
            medoids = np.vstack([medoids, max_distance_point])

        clusters = [[] for _ in range(len(medoids))]

        # Distribution of points to the nearest medoids
        for idx, point in enumerate(points):
            distances = np.linalg.norm(medoids - point, axis=1)
            nearest_medoid_idx = np.argmin(distances)
            clusters[nearest_medoid_idx].append(idx)

        need_continue = False
        max_distance_point = None
        max_average_distance = -1

        # Updating the medoids for each cluster
        for i in range(len(clusters)):
            if clusters[i]:
                cluster_indx = clusters[i]
                cluster_points = points[cluster_indx]


                min_total_distance = float('inf')
                max_total_distance = -1
                max_point = None
                new_medoid = None

                for point in cluster_points:
                    total_distance = np.sum(np.linalg.norm(cluster_points-point, axis=1))

                    if total_distance < min_total_distance:
                        min_total_distance = total_distance
                        new_medoid = point
                    if total_distance > max_total_distance:
                        max_total_distance = total_distance
                        max_point = point

                # Calculate the average distance
                average_distance = min_total_distance / len(cluster_points)

                # Check if we need to add new medoid
                if average_distance > MAX_AVERAGE_DISTANCE:
                    need_continue = True
                    max_distance_point = max_point

                if average_distance > max_average_distance:
                    max_average_distance = average_distance

                medoids[i] = new_medoid

        print(f"Iteration {iteration}: Max average distance = {max_average_distance}")
        iteration += 1

        if (len(clusters) > 1 and STEP_BY_STEP_VISUALIZATION):
            visualisation(clusters, medoids)

    clusters_coords = [[points[idx] for idx in cluster] for cluster in clusters]
    return medoids, clusters_coords


def divisive_clustering_centroid(points):
    clusters = [points]
    centroids = [np.mean(points, axis=0)]
    need_continue = True
    iteration = 1


    while need_continue:
        max_average_distance = -1
        need_continue = False
        cluster_to_split = None

        # Find the cluster with the largest average distance to the centroid
        for i, cluster in enumerate(clusters):
            if len(cluster) > 1:
                centroid = centroids[i]

                distances = [np.linalg.norm(point - centroid) for point in cluster]
                average_distance = np.mean(distances)

                if average_distance > max_average_distance:
                    max_average_distance = average_distance
                    cluster_to_split = i

        # Check to see if we need to continue the split
        if max_average_distance > MAX_AVERAGE_DISTANCE:
            need_continue = True

            # Split the selected cluster into two
            points_to_split = np.array(clusters[cluster_to_split])

            # kmeans_indices = np.random.choice(points_to_split.shape[0], 2, replace=False)
            # centroids_to_split = points_to_split[kmeans_indices]

            distances = np.linalg.norm(points_to_split - centroids[cluster_to_split], axis=1)
            idx1 = np.argmax(distances)
            distances = np.linalg.norm(points_to_split - points_to_split[idx1], axis=1)
            idx2 = np.argmax(distances)
            centroids_to_split = np.array([points_to_split[idx1], points_to_split[idx2]])

            # Perform partitioning into two clusters
            new_clusters = [[], []]
            for point in points_to_split:
                distances = [np.linalg.norm(point - centroid) for centroid in centroids_to_split]
                nearest_centroid_idx = np.argmin(distances)
                new_clusters[nearest_centroid_idx].append(point)

            # Update the centroids
            new_centroids = [np.mean(cluster, axis=0) for cluster in new_clusters]

            # Replace clusters
            clusters.pop(cluster_to_split)
            centroids.pop(cluster_to_split)

            clusters.extend(new_clusters)
            centroids.extend(new_centroids)

        print(f"Iteration {iteration}: Number of clusters = {len(clusters)}, Max average distance = {max_average_distance}")
        iteration += 1
        if (len(clusters) > 1 and STEP_BY_STEP_VISUALIZATION):
            visualisation(clusters, centroids)

    return centroids, clusters


def main():
    point_array = generate_points()
    if (MODE == 1):
        centroids, clusters = knn_clustering_centroid(point_array)
    elif (MODE == 2):
        centroids, clusters = knn_clustering_medoid(point_array)
    else:
        centroids, clusters = divisive_clustering_centroid(point_array)


    # visualization
    try:
        visualisation(clusters, centroids)
    except Exception:
        pass


if __name__ == '__main__':
    main()







