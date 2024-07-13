import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

# Declare constants
R0 = np.identity(3)
t0 = np.zeros(3)
num_ICP_iters = 30
dmax = 0.25

# Load input points
X = np.loadtxt("pclX.txt")
Y = np.loadtxt("pclY.txt")

def EstimateCorrespondences(X, Y, t, R, dmax):
    C = []  # List to store correspondences
    for i, xi in enumerate(X):
        # Apply transformation to xi
        transformed_xi = np.dot(R, xi) + t
        # Find the closest point yj in Y
        distances = distance.cdist([transformed_xi], Y, 'euclidean')[0]
        closest_idx = np.argmin(distances)
        # Check if distance is within threshold
        if distances[closest_idx] < dmax:
            C.append((i, closest_idx))
    return C

def ComputeOptimalRigidRegistration(X, Y, C):
    # Get correspondences from C array
    X_correspondences = np.array([X[i] for i, _ in C])
    Y_correspondences = np.array([Y[j] for _, j in C])
    # Calculate pointcloud centroids
    centroid_x = np.mean(X_correspondences, axis=0)
    centroid_y = np.mean(Y_correspondences, axis=0)
    # Calculate deviations
    X_deviation = X_correspondences - centroid_x
    Y_deviation = Y_correspondences - centroid_y
    # Compute cross-covariance matrix
    W = X_deviation.T @ Y_deviation / len(C)
    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(W)
    # Construct Optimal Rotation
    V = Vt.T
    d = np.linalg.det(V @ U.T)
    M = np.diag([1, 1, d])
    R = V @ M @ U.T
    # Recover Optimal Translation
    t = centroid_y - R @ centroid_x
    return t, R

def ICP(X, Y, t0, R0, dmax, num_ICP_iters):
    t = t0
    R = R0
    for _ in range(num_ICP_iters):
        # Estimate correspondences
        C = EstimateCorrespondences(X, Y, t, R, dmax)
        # Compute optimal rigid registration
        t, R = ComputeOptimalRigidRegistration(X, Y, C)
    return t, R

if __name__ == "__main__":
    # Apply ICP
    t_final, R_final = ICP(X, Y, t0, R0, dmax, num_ICP_iters)
    # Transform X using the final t and R
    X_transformed = np.dot(X, R_final.T) + t_final
    
    # Plot figures
    fig = plt.figure(figsize=(14, 7))
    
    # Original Point Clouds
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', s=1, label='X Original')
    ax1.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='r', s=1, label='Y Original')
    ax1.set_title('Original Point Clouds')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=0, azim=0)
    ax1.legend()
    
    # Transformed Point Cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='r', s=1, label='Y')
    ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c='b', s=1, label='X Transformed')
    ax2.set_title('Transformed Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=0, azim=0)
    ax2.legend()
    
    plt.show()
    
    # Compute RMSE
    C_final = EstimateCorrespondences(X, Y, t_final, R_final, dmax)
    errors = [np.linalg.norm(Y[j] - (np.dot(X[i], R_final.T) + t_final)) for i, j in C_final]
    RMSE = np.sqrt(np.mean(np.square(errors)))
    print("RMSE:", RMSE)
    print("R:", R_final)
    print("t:", t_final)
