import numpy as np

def mean_face(faces):
    return np.mean(faces, axis=0)

def center_faces(faces, mean_face_vector):
    return faces - mean_face_vector

def covariance_matrix(X):
    return np.dot(X.T, X) / (X.shape[0] - 1)

def eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(-eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]

def project_face(face, mean_face_vector, eigenvectors, k):
    centered = face - mean_face_vector
    return np.dot(centered, eigenvectors[:, :k])

def reconstruct_face(projection, mean_face_vector, eigenvectors, k):
    return np.dot(projection, eigenvectors[:, :k].T) + mean_face_vector

