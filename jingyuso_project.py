## Maintainer: Jingyu Song #####
## Contact: jingyuso@umich.edu #####

import numpy as np
import os
from scipy.spatial import KDTree
import h5py
import open3d as o3d
from sklearn import cluster
import random
import time
from scipy.spatial.transform import Rotation


try:
    import open3d as o3d
    visualize = True
except ImportError:
    print('To visualize you need to install Open3D. \n \t>> You can use "$ pip install open3d"')
    visualize = False

from assignment_4_helper import ICPVisualizer, load_point_cloud, view_point_cloud, quaternion_matrix, \
    quaternion_from_axis_angle, load_pcs_and_camera_poses, save_point_cloud


def transform_point_cloud(point_cloud, t, R):
    """
    Transform a point cloud applying a rotation and a translation
    :param point_cloud: np.arrays of size (N, 6)
    :param t: np.array of size (3,) representing a translation.
    :param R: np.array of size (3,3) representing a 3D rotation matrix.
    :return: np.array of size (N,6) resulting in applying the transformation (t,R) on the point cloud point_cloud.
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    # print(R @ point_cloud[:,:3].T)

    transformed_point_cloud = (R @ point_cloud[:, 0:3].T + t.reshape((3,1))).T
    # transformed_point_cloud = np.hstack((transformed_point_cloud, point_cloud[:, 3:6]))

    # transformed_point_cloud = None  # TODO: Replace None with your result
    # ------------------------------------------------
    return transformed_point_cloud


def compute_transform_error(pcA, pcB):
    temp = np.mean(np.linalg.norm(pcA - pcB, axis=1))
    return temp

def xyz_to_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def generate_random_transform(sample_list, is_simple=True):
    if is_simple:
        num_sample = len(sample_list)
        trans = 2*num_sample * np.random.rand(num_sample,3)
        trans[:,1] = 0
        ori = np.random.rand(num_sample,1) * 2 * np.pi
        # ori = np.zeros((num_sample,1))
        # ori = np.hstack((ori,np.zeros((num_sample,2))))
        ori_R = Rotation.from_euler('y', ori)
        ori_q = ori_R.as_quat()
    else:
        num_sample = len(sample_list)
        trans = 2*num_sample * np.random.rand(num_sample,3)
        ori = np.random.rand(num_sample,3) * 2 * np.pi
        ori_R = Rotation.from_euler('yzx', ori)
        ori_q = ori_R.as_quat()
    # ori = 2 * np.random.rand(num_sample,4) - 1
    return (trans, ori_q)

def icp_step(point_cloud_A, point_cloud_B, t_init, R_init):
    """
    Perform an ICP iteration to find a new estimate of the pose of the model point cloud with respect to the scene pointcloud.
    :param point_cloud_A: np.array of size (N_a, 6) (scene)
    :param point_cloud_B: np.array of size (N_b, 6) (model)
    :param t_init: np.array of size (3,) representing the initial transformation candidate
                    * It may be the output from the previous iteration
    :param R_init: np.array of size (3,3) representing the initial rotation candidate
                    * It may be the output from the previous iteration
    :return:
        - t: np.array of size (3,) representing a translation estimate between point_cloud_A and point_cloud_B
        - R: np.array of size (3,3) representing a 3D rotation estimate between point_cloud_A and point_cloud_B
        - correspondences: np.array of size(n_a,) containing the closest point indexes in point_cloud_B
            for each point in point_cloud_A
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    transformed_point_B = transform_point_cloud(point_cloud_B, t_init, R_init)
    correspondences = find_closest_points(point_cloud_A, transformed_point_B)
    
    t, R = find_best_transform(point_cloud_A, transformed_point_B[correspondences])
    new_transform = np.eye(4)
    prev_transform = np.eye(4)
    new_transform[:3,:3] = R
    new_transform[:3,3] = t
    prev_transform[:3,:3] = R_init
    prev_transform[:3,3] = t_init
    final_transform = new_transform @ prev_transform
    R = final_transform[:3,:3]
    t = final_transform[:3,3]

    # t = None    # TODO: Replace None with your result
    # R = None    # TODO: Replace None with your result
    # correspondences = None  # TODO: Replace None with your result
    # ------------------------------------------------
    return t, R, correspondences

def find_best_transform(point_cloud_A, point_cloud_B):
    """
    Find the transformation 2 corresponded point clouds.
    Note 1: We assume that each point in the point_cloud_A is corresponded to the point in point_cloud_B at the same location.
        i.e. point_cloud_A[i] is corresponded to point_cloud_B[i] forall 0<=i<N
    :param point_cloud_A: np.array of size (N, 6) (scene)
    :param point_cloud_B: np.array of size (N, 6) (model)
    :return:
         - t: np.array of size (3,) representing a translation between point_cloud_A and point_cloud_B
         - R: np.array of size (3,3) representing a 3D rotation between point_cloud_A and point_cloud_B
    Note 2: We transform the model to match the scene.
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    
    # we only need the position of the pointcloud
    point_cloud_A = point_cloud_A[:,:3]
    point_cloud_B = point_cloud_B[:,:3]
    mu_A = np.mean(point_cloud_A, axis=0)
    mu_B = np.mean(point_cloud_B, axis=0)
    # W = np.zeros((3,3))
    W = (point_cloud_A - mu_A).T @ ((point_cloud_B - mu_B))
    # for i in range(point_cloud_A.shape[0]):
    #     W = W + (point_cloud_A[i,:] - mu_A) @ ((point_cloud_B[i,:] - mu_B).T)
    U, D, V = np.linalg.svd(W)
    # temp = np.eye(V.shape[1])
    # temp[-1,-1] = np.linalg.det(V.T @ U.T)
    R = U @ V
    t = mu_A - R @ mu_B

    # t = None    # TODO: Replace None with your result
    # R = None    # TODO: Replace None with your result
    # ------------------------------------------------
    return t, R

def find_closest_points(point_cloud_A, point_cloud_B):
    """
    Find the closest point in point_cloud_B for each element in point_cloud_A.
    :param point_cloud_A: np.array of size (n_a, 6)
    :param point_cloud_B: np.array of size (n_b, 6)
    :return: np.array of size(n_a,) containing the closest point indexes in point_cloud_B
            for each point in point_cloud_A
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    tree = KDTree(point_cloud_B[:,:3])
    n_a = point_cloud_A.shape[0]
    closest_points_indxs = np.zeros(n_a, dtype=int)
    # for i in range(n_a):
    _, closest_points_indxs = tree.query(point_cloud_A[:,:3], k=1)


    # closest_points_indxs = None # TODO: Replace None with your result
    # ------------------------------------------------
    return closest_points_indxs

def icp(point_cloud_A, point_cloud_B, num_iterations=50, t_init=None, R_init=None, visualize=True):
    """
    Find the
    :param point_cloud_A: np.array of size (N_a, 6) (scene)
    :param point_cloud_B: np.array of size (N_b, 6) (model)
    :param num_iterations: <int> number of icp iteration to be performed
    :param t_init: np.array of size (3,) representing the initial transformation candidate
    :param R_init: np.array of size (3,3) representing the initial rotation candidate
    :param visualize: <bool> Whether to visualize the result
    :return:
         - t: np.array of size (3,) representing a translation estimate between point_cloud_A and point_cloud_B
         - R: np.array of size (3,3) representing a 3D rotation estimate between point_cloud_A and point_cloud_B
    """
    if t_init is None:
        t_init = np.zeros(3)
    if R_init is None:
        R_init = np.eye(3)
    if visualize:
        vis = ICPVisualizer(point_cloud_A, point_cloud_B)
    t = t_init
    R = R_init
    correspondences = None  # Initialization waiting for a value to be assigned
    if visualize:
        vis.view_icp(R=R, t=t)
    for i in range(num_iterations):
        # ------------------------------------------------
        # FILL WITH YOUR CODE
        t, R, correspondences = icp_step(point_cloud_A, point_cloud_B, t, R)
        # t = None    # TODO: Replace None with your result
        # R = None  # TODO: Replace None with your result
        # correspondences = None  # TODO: Replace None with your result
        # ------------------------------------------------
        if visualize:
            vis.plot_correspondences(correspondences)   # Visualize point correspondences
            time.sleep(.005)  # Wait so we can visualize the correspondences
            vis.view_icp(R, t)  # Visualize icp iteration

    return t, R

def perfect_model_icp(pcB, pcA, visualize=True):
    # Load the model
    # pcB = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Model
    pcB = np.hstack((pcB, np.array([.73, .21, .1]) * np.ones((pcB.shape[0], 3))))  # Paint it red
    pcA = np.hstack((pcA, np.array([.01, .01, .1]) * np.ones((pcB.shape[0], 3))))  # Paint it black
    # pcA = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Perfect scene
    # Apply transfomation to scene so they differ
    # t_gth = np.array([0.4, -0.2, 0.2])
    # r_angle = np.pi / 2
    # R_gth = quaternion_matrix(np.array([np.cos(r_angle / 2), 0, np.sin(r_angle / 2), 0]))
    # pcA = transform_point_cloud(pcA, R=R_gth, t=t_gth)
    R_init = quaternion_matrix([0,0,0,1])
    t_init = np.mean(pcA[:, :3], axis=0)

    # ICP -----
    t, R = icp(pcA, pcB, num_iterations=70, t_init=t_init, R_init=R_init, visualize=visualize)
    transformed_point_B = transform_point_cloud(pcB, t, R)
    correspondences = find_closest_points(pcA, transformed_point_B)
    error = compute_transform_error(pcA[:,:3], transformed_point_B[correspondences])
    print('Infered Position: ', t)
    print('Infered Orientation:', R)
    if error < 0.01:
        return True
    else:
        return False
    # print('\tReal Position: ', t_gth)
    # print('\tReal Orientation:', R_gth)

if __name__ == '__main__':
    # we have all the models ./models folder
    model_dir = './models'

    models = [os.path.splitext(filename)[0] for filename in os.listdir(model_dir)]
    model_list = []
    for model in models:     
        model_list.append(np.load(os.path.join(model_dir, f'{model}.npy')))
    num_models = len(model_list)

    # select a random number between 2 and num_models
    num_objects = np.random.randint(2, num_models+1)
    sample_list = model_list
    random.shuffle(sample_list)
    sample_list = sample_list[:num_objects]
    transformed_samples = []
    transformed_samples_pcd = []

    # the sample list contains the sample of objects,
    # we then transform it to different places
    trans, ori = generate_random_transform(sample_list, is_simple=False)
    for i in range(num_objects):
        transformed_pc = transform_point_cloud(sample_list[i], trans[i,:], quaternion_matrix(ori[i,:]))
        transformed_samples.append(transformed_pc)
        transformed_samples_pcd.append(xyz_to_pcd(transformed_pc))
    o3d.visualization.draw_geometries(transformed_samples_pcd)

    transformed_pc_arr = np.array(transformed_samples).reshape(-1,3)
    kmeans = cluster.KMeans(n_clusters=num_objects).fit(transformed_pc_arr)
    count = 0
    for i in range(num_objects):
        t_gt = trans[i,:]
        o_gt = ori[i,:]
        pcA = transformed_pc_arr[kmeans.labels_==i]
        print('\tReal Position: ', t_gt)
        print('\tReal Orientation:', o_gt)
        for j in range(num_models):
            pcB = model_list[j]
            if perfect_model_icp(pcB, pcA, visualize=False):
                count += 1
    print(f'correct {count} times in total {num_objects}')
            


    # # here we use two models
    
    # pcd_list = []
    # model0 = np.load(os.path.join(model_dir, f'{models[0]}.npy'))
    # model1 = np.load(os.path.join(model_dir, f'{models[1]}.npy'))

    # pcd0 = o3d.geometry.PointCloud()
    # pcd1 = o3d.geometry.PointCloud()

    # pcd0.points = o3d.utility.Vector3dVector(model0)
    # pcd1.points = o3d.utility.Vector3dVector(model1)
    # pcd_list = [pcd0, pcd1]
    # # o3d.visualization.draw_geometries(pcd_list)

    # # vis_xyz_array(model0)
    # # vis_xyz_array(model1)

    # pcd_list = []

    # r_angle = np.pi / 2
    # model0_tr = transform_point_cloud(model0, np.array([1,1,1]), quaternion_matrix(np.array([np.cos(r_angle / 2), 0, np.sin(r_angle / 2), 0])))
    # model1_tr = transform_point_cloud(model1, np.array([-1,-1,-1]), quaternion_matrix(np.array([np.cos(r_angle / 2), 0, np.sin(r_angle / 2), 0])))
    # # assume we have the models at the origin and then transform it to 
    # pcd0_tr = xyz_to_pcd(model0_tr)
    # pcd1_tr = xyz_to_pcd(model1_tr)

    # models_together = np.vstack((model0_tr, model1_tr))
    # print(models_together.shape)
    # o3d.visualization.draw_geometries([pcd0_tr, pcd1_tr])
    # kmeans = cluster.KMeans(n_clusters=2).fit(models_together)
    # print(kmeans)