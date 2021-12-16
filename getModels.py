## Maintainer: Jingyu Song #####
## Contact: jingyuso@umich.edu #####

import numpy as np
import os
from scipy.spatial import KDTree
import h5py
import open3d as o3d

if __name__ == '__main__':
    hf = h5py.File('./modelnet10/train0.h5', 'r')
    data = hf['data'][:].astype('float32')
    label = hf['label'][:].astype('int64')

    # get the models
    model_list = [0,1,2,3,5]
    name_list = ["table", "monitor", "chair", "bed", "sofa"]

    for idx in range(5):
        xyz = data[idx,:,:]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        np.save(f'./models/{name_list[idx]}.npy', xyz)
        o3d.io.write_point_cloud(f'./models/{name_list[idx]}.ply', pcd)

    
    

    


    
    # pcd_load = o3d.io.read_point_cloud("./sync.ply")
    # o3d.visualization.draw_geometries([pcd_load])
    # o3d.visualization.draw_geometries([xyz])
    # print(data)

    # data = hf.get('dataset_name').value
