import random
import numpy as np
import torch


#src: https://github.com/zhenxianglance/PCBA/blob/main/dataset/dataset.py
def center_and_scale(points: np.array) -> np.array:
    points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale
    return points #returns points in unit sphere

def random_point_in_cube() -> np.array: #random points in cube
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    z = random.uniform(-1, 1)
    return np.array([x,y,z], dtype=np.float32)

def create_sample_pointcloud_for_tensor(N: int) -> np.array:
    pointcloud = []
    for n in range(N):
        pointcloud.append(random_point_in_cube())
    pointcloud = np.array(pointcloud, dtype=np.float32)
    return pointcloud

def transpose_and_batch_pointcloud_for_tensor(pointcloud: np.array, B: int, N: int) -> np.array:
    batch = []
    for b in range(B):
        batch.append(pointcloud)
    batch = np.array(batch, dtype=np.float32)
    #(B, N, 3)->(B, 3, N)
    batch = np.transpose(batch, (0, 2, 1))
    return batch

def gen_tensor_from_pointcloud(pointcloud: np.array) -> torch.FloatTensor:
    return torch.FloatTensor(pointcloud)

def is_in_cube(point: np.array, cube: np.array, stepsize: float) -> bool: #cube: x_end, y_end, z_end
    x = point[0]
    y = point[1]
    z = point[2]
    x_end = cube[0]
    y_end = cube[1]
    z_end = cube[2]
    x_start = x_end - stepsize
    y_start = y_end - stepsize
    z_start = z_end - stepsize

    return ((x_start <= x < x_end)
            and (y_start <= y < y_end)
            and (z_start <= z < z_end))

def generate_cubes(stepsize:float) -> np.array:
    cube_list = []
    for x in np.arange(-1, 1, stepsize, dtype = np.float32): #excluding upper range
        for y in np.arange(-1, 1, stepsize, dtype = np.float32):
            for z in np.arange (-1, 1, stepsize, dtype = np.float32):
                cube_list.append([x+stepsize,y+stepsize,z+stepsize]) #include upper range by only saving upper range
    return cube_list

def choose_sub_pointclouds(pointcloud: np.array, granularity: int) -> np.array:
    sub_pointclouds = [] #collects the indices of the points that are in each cube
    stepsize = 2 / granularity #granularity as percentage (max. 1/100)
    cube_list = generate_cubes(stepsize)
    for cube in cube_list:
        index_list = []
        for i in range(len(pointcloud)):
            if is_in_cube(pointcloud[i], cube, stepsize):
                index_list.append(i) #create index list such that subpointclouds looks like: [[i_a, i_b, i_c], [i_d, i_e],...]
        sub_pointclouds.append(index_list)
    return sub_pointclouds

def perturb_point(point: np.array, max_perturbation) -> np.array:
    x_temp = point[0] + random.uniform(-max_perturbation, max_perturbation)
    y_temp = point[1] + random.uniform(-max_perturbation, max_perturbation)
    z_temp = point[2] + random.uniform(-max_perturbation, max_perturbation)
    temp_point = np.array([x_temp, y_temp, z_temp])
    return temp_point

def perturb_points_in_cube(pointcloud: np.array, points_in_subcube: list, cube: np.array, stepsize:float) -> np.array: #returns pointcloud with perturbed points
    perturbed_pointcloud = pointcloud.copy()
    max_perturbation = stepsize / 10
    for i in range(len(points_in_subcube)):
        point_index = points_in_subcube[i]
        point = pointcloud[point_index]
        for _ in range(10): #maximum of 10 tries for each point to be within box-bounds after perturbation
            perturbed_point = perturb_point(point, max_perturbation)
            if is_in_cube(perturbed_point, cube, stepsize):
                perturbed_pointcloud[point_index] = perturbed_point
                break
    return perturbed_pointcloud