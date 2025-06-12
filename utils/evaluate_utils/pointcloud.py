#!/usr/bin/env python3

import logging
import math
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import trimesh
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool

def create_pc_fast(
    model, shape_feature, N=256, max_batch=1000000, level_set=0.0, from_plane_features=False
):    
    model.eval()
    oc_list = [2 **4, 2 **2, 2 **2,  2 **2]
    pred_res = 1
    for i in range(len(oc_list)):
        res = oc_list[i]
        pred_res = pred_res * res
        size = 2.0 / pred_res
        if i == 0:
            x_start = -1.0
            y_start = -1.0
            z_start = -1.0
            cube = create_cube_fast(res, size, x_start, y_start, z_start)
            cube_points = cube.shape[0]
            head = 0
            while head < cube_points:
                query = cube[head : min(head + max_batch, cube_points), 0:3].unsqueeze(0)
                if from_plane_features:
                    pred_occ = model.forward_with_plane_features_occ(shape_feature.cuda(), query.cuda()).detach().cpu()
                else:
                    pred_occ = model(shape_feature.cuda(), query.cuda()).detach().cpu()
                cube[head : min(head + max_batch, cube_points), 3] = pred_occ.squeeze()
                head += max_batch
            sorted_indices = torch.argsort(cube[:, -1])
            sorted_cube = cube[sorted_indices]
            sorted_cube = sorted_cube[int(0.8 * sorted_cube.shape[0]):,:3]
        else:
            new_cobes = []
            for j in range(sorted_cube.shape[0]):
                x_start = sorted_cube[j,0] - 1.0 / (pred_res / res)
                y_start = sorted_cube[j,1] - 1.0 / (pred_res / res)
                z_start = sorted_cube[j,2] - 1.0 / (pred_res / res)
                cube = create_cube_fast(res, size, x_start, y_start, z_start)
                new_cobes.append(cube)
            cube = torch.cat(new_cobes, 0)
            cube_points = cube.shape[0]
            head = 0
            while head < cube_points:
                query = cube[head : min(head + max_batch, cube_points), 0:3].unsqueeze(0)
                if from_plane_features:
                    pred_occ = model.forward_with_plane_features_occ(shape_feature.cuda(), query.cuda()).detach().cpu()
                else:
                    pred_occ = model(shape_feature.cuda(), query.cuda()).detach().cpu()
                cube[head : min(head + max_batch, cube_points), 3] = pred_occ.squeeze()
                head += max_batch
            sorted_indices = torch.argsort(cube[:, -1])
            sorted_cube = cube[sorted_indices]
            if i < len(oc_list) - 1:
                sorted_cube = sorted_cube[int(0.8 * sorted_cube.shape[0]):,:3]
    return sorted_cube[-650000:,:3].reshape(1, -1, 3).cuda()
        

  

def create_cube_fast(N, size, x_start, y_start, z_start):
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)
    voxel_origin = [x_start, y_start, z_start]
    voxel_size = size

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long().float() / N) % N
    samples[:, 0] = ((overall_index.long().float() / N) / N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0] + voxel_size / 2 
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1] + voxel_size / 2 
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2] + voxel_size / 2 

    samples.requires_grad = False

    return samples

def create_cube(N):

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long().float() // N) % N
    samples[:, 0] = ((overall_index.long().float() // N) // N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False

    return samples

def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12) 
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

    return dis_idx

def create_pc_optimizer(
    model, shape_feature, points=None
):
    
    learning_rate = 0.00001
    epochs = 50

    
    if points == None:
        points = generate_points_on_sphere(N=100000)    
    points.requires_grad = True
    points = torch.nn.Parameter(points)

    model.eval()
    optimizer = torch.optim.Adam([points], lr=learning_rate)
    for epoch in range(epochs):

        optimizer.zero_grad()
        pred_occ = model.forward_with_plane_features_occ(shape_feature.cuda(), points)

        pred_loss = -1 * torch.mean(pred_occ)

        loss = pred_loss

        loss.backward()

        optimizer.step()
       
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, pred {pred_loss.item()}')
    return points.reshape(1, -1, 3)

def pc_optimizer(model, shape_feature, points):
    num_points = points.shape[1]
    dis = (torch.rand(num_points, 3) - 0.5) * 0.002
    learning_rate = 0.00001
    epochs = 20
    dis.requires_grad = True
    points.requires_grad = False
    dis = dis.cuda()
    dis = torch.nn.Parameter(dis)
    model.eval()
    optimizer = torch.optim.Adam([dis], lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_occ = model.forward_with_plane_features_occ(shape_feature.cuda(), points + dis)
        pred_loss = -1 * torch.mean(pred_occ)
        loss = pred_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        dis = torch.clamp(dis, max=0.001, min=-0.001)
        dis = torch.nn.Parameter(dis)
        optimizer = torch.optim.Adam([dis], lr=learning_rate)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, pred {pred_loss.item()}')
    dis = torch.clamp(dis, max=0.001, min=-0.001)
    return dis + points


def generate_points_on_sphere(N=2048, R=0.25):
    theta = np.random.uniform(0, 2*np.pi, N)
    z = np.random.uniform(-1, 1, N)
    phi = np.arccos(z)
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * z
    pc = torch.from_numpy(np.vstack((x, y, z)).T).reshape(1, -1, 3).float().cuda()

    return pc 


def clean_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points.reshape(-1, 3)))

    nb_neighbors=20
    std_ratio=2.0
    clean_cloud, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    nb_points=16
    radius=0.05
    clean_cloud, ind = clean_cloud.remove_radius_outlier(nb_points, radius)

    return np.asarray(clean_cloud.points)

def mls_smoothing(points, k=1000):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    smoothed_points = np.copy(points)
    
    for i, (point, idx) in enumerate(zip(points, indices)):
        neighbors = points[idx]
        A = np.c_[neighbors[:, 0], neighbors[:, 1], np.ones(len(neighbors))]
        b = neighbors[:, 2]
        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        smoothed_points[i, 2] = coef[0] * point[0] + coef[1] * point[1] + coef[2]
    
    return smoothed_points

def gaussian_filter(points, radius=0.2, sigma=0.05):
    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(points)
    indices = nbrs.radius_neighbors(points, return_distance=False)
    
    smoothed_points = np.zeros_like(points)
    
    for i, idx in enumerate(indices):
        print(i)
        neighbors = points[idx]
        distances = np.linalg.norm(neighbors - points[i], axis=1)
        weights = np.exp(-0.5 * (distances / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        weights /= weights.sum()
        smoothed_points[i] = np.sum(neighbors * weights[:, np.newaxis], axis=0)
    
    return smoothed_points


def process_point(i, indices, points, sigma):
    print(i)
    neighbors = points[indices[i]]
    distances = np.linalg.norm(neighbors - points[i], axis=1)
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    weights /= weights.sum()
    return np.sum(neighbors * weights[:, np.newaxis], axis=0)

def gaussian_filter_parallel(points, radius=0.2, sigma=0.05, num_processes=10):
    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(points)
    indices = nbrs.radius_neighbors(points, return_distance=False)
    
    with Pool(num_processes) as pool:
        args = [(i, indices, points, sigma) for i in range(len(points))]
        smoothed_points = np.array(pool.starmap(process_point, args))
    
    return smoothed_points

if __name__ == '__main__':
    cube = create_cube_fast(2, 1.0, 0.0, 0.0, 0.0)
    
    pass
