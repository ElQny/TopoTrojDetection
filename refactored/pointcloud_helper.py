import random
import numpy as np
import torch


#src: https://github.com/zhenxianglance/PCBA/blob/main/dataset/dataset.py
def center_and_scale(points: np.array) -> np.array:
    """
       Uses a pointcloud to center and scale to [-1, 1] in all dimensions
       This is done because PCBA expects a unit sphere of points
    """
    print("Centering and scaling pointcloud")
    points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale
    return points #returns points in unit sphere

def is_in_unit_sphere(point: np.array) -> bool:
    """
       Checks if a point is inside the unit sphere by calculating the norm
       r = sqrt(x*x + y*y + z*z)
    """
    x, y, z = point
    return np.sqrt(x*x + y*y + z*z) <= 1.0

def random_point_in_unit_sphere() -> np.array: #random points
    while True:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform(-1, 1)
        point = np.array([x, y, z], dtype=np.float32)
        if is_in_unit_sphere(point):
            return point

def create_sample_pointcloud(N: int) -> np.array:
    print("Creating sample pointcloud")
    pointcloud = []
    for n in range(N):
        pointcloud.append(random_point_in_unit_sphere())
    pointcloud = np.array(pointcloud, dtype=np.float32)
    return pointcloud

def is_in_cube(point: np.array, cube: np.array, granularity: int) -> bool: #cube: x_end, y_end, z_end
    stepsize = calc_stepsize(granularity)
    x, y, z = point
    x_end, y_end, z_end = cube
    x_start = x_end - stepsize
    y_start = y_end - stepsize
    z_start = z_end - stepsize

    return ((x_start <= x < x_end) #ensuring all points are in one cube (one border is inclusive)
            and (y_start <= y < y_end)
            and (z_start <= z < z_end))


def generate_cubes(granularity: int) -> np.array:
    print("Generating cubes")
    stepsize = calc_stepsize(granularity)
    cube_list = []
    for x in np.arange(-1, 1, stepsize, dtype = np.float32): #excluding upper range
        for y in np.arange(-1, 1, stepsize, dtype = np.float32):
            for z in np.arange (-1, 1, stepsize, dtype = np.float32):
                cube_list.append([x+stepsize,y+stepsize,z+stepsize]) #include upper range by only saving upper range
    return cube_list

def choose_sub_pointclouds(pointcloud: np.array, granularity: int) -> list:
    print("Choosing sub pointclouds")
    cube_list = generate_cubes(granularity)
    sub_pointclouds = [] #collects the indices of the points that are in each cube

    for cube in cube_list:
        index_list = []
        for i in range(len(pointcloud)):
            if is_in_cube(pointcloud[i], cube, granularity):
                index_list.append(i) #create index list such that subpointclouds looks like: [[i_a, i_b, i_c], [i_d, i_e],...]
        sub_pointclouds.append(index_list)

    return sub_pointclouds

def perturb_point(point: np.array, max_perturbation) -> np.array:
    x_temp = point[0] + random.uniform(-max_perturbation, max_perturbation)
    y_temp = point[1] + random.uniform(-max_perturbation, max_perturbation)
    z_temp = point[2] + random.uniform(-max_perturbation, max_perturbation)
    temp_point = np.array([x_temp, y_temp, z_temp])
    return temp_point

def perturb_points_in_cube(
        pointcloud: np.array,
        points_in_subcube: list,
        cube: np.array,
        granularity:int
) -> np.array: #returns pointcloud with perturbed points

    return pointcloud.copy() #for testing
    # perturbed_pointcloud = pointcloud.copy()
    # max_perturbation = calc_stepsize(granularity) / 10
    #
    # for point_index in points_in_subcube:
    #     point = pointcloud[point_index]
    #
    #     for _ in range(10): #maximum of 10 tries for each point to be within box-bounds after perturbation
    #         perturbed_point = perturb_point(point, max_perturbation)
    #         if is_in_cube(perturbed_point, cube, granularity) and is_in_unit_sphere(perturbed_point):
    #             perturbed_pointcloud[point_index] = perturbed_point
    #             break
    # return perturbed_pointcloud

def transpose_and_batch_pointclouds_to_tensor(pointclouds: np.array) -> torch.FloatTensor:
    #(B, N, 3)->(B, 3, N)
    batch = np.transpose(pointclouds, (0, 2, 1))
    return torch.FloatTensor(batch)

def calc_stepsize(granularity: int) -> np.float32:
    return 2 / granularity