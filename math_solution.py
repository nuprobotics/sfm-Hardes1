import numpy as np

def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    Computes 3D points via triangulation from two camera views.

    :param camera_matrix: Camera intrinsic matrix (3x3).
    :param camera_position1: First camera position in world coordinates (3x1).
    :param camera_rotation1: First camera rotation matrix in world coordinates (3x3).
    :param camera_position2: Second camera position in world coordinates (3x1).
    :param camera_rotation2: Second camera rotation matrix in world coordinates (3x3).
    :param image_points1: Points in the first image (Nx2).
    :param image_points2: Points in the second image (Nx2).
    :return: Triangulated 3D points (Nx3).
    """

    # Ensure positions are column vectors
    camera_position1 = camera_position1.reshape((3, 1))
    camera_position2 = camera_position2.reshape((3, 1))

    # Compute rotation matrices from world to camera coordinates
    R1 = camera_rotation1.T
    R2 = camera_rotation2.T

    # Compute translation vectors
    t1 = -R1 @ camera_position1
    t2 = -R2 @ camera_position2

    # Form the extrinsic matrices
    extrinsic_matrix1 = np.hstack((R1, t1))
    extrinsic_matrix2 = np.hstack((R2, t2))

    # Compute projection matrices
    projection_matrix1 = camera_matrix @ extrinsic_matrix1
    projection_matrix2 = camera_matrix @ extrinsic_matrix2

    # Triangulate points
    triangulated_points = []
    for p1, p2 in zip(image_points1, image_points2):
        # Construct the linear system
        A = np.vstack([
            p1[0] * projection_matrix1[2, :] - projection_matrix1[0, :],
            p1[1] * projection_matrix1[2, :] - projection_matrix1[1, :],
            p2[0] * projection_matrix2[2, :] - projection_matrix2[0, :],
            p2[1] * projection_matrix2[2, :] - projection_matrix2[1, :]
        ])

        # Solve for the 3D point using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]  # Normalize to convert from homogeneous coordinates

        triangulated_points.append(X[:3])

    return np.array(triangulated_points)