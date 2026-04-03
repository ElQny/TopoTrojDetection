import random
import numpy as np
import torch


#src: https://github.com/zhenxianglance/PCBA/blob/main/dataset/dataset.py
def center_and_scale(points: np.array) -> np.array:
    points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale
    return points #returns points in unit sphere

def random_point_in_unit_sphere() -> np.array:
    x, y, z = 0, 0, 0
    while(True):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform(-1, 1)
        if ((x*x + y*y + z*z) <= 1.0):
            break
    return np.array([x,y,z], dtype=np.float32)

def create_sample_pointcloud_for_tensor(B: int, N: int) -> np.array:
    batch = []
    for b in range(B):
        pointcloud = []
        for n in range(N):
            pointcloud.append(random_point_in_unit_sphere())
        pointcloud = np.array(pointcloud, dtype=np.float32)
        pointcloud = center_and_scale(pointcloud)
        batch.append(pointcloud)
    batch = np.array(batch, dtype=np.float32)
    #(B, N, 3)->(B, 3, N)
    batch = np.transpose(batch, (0, 2, 1))
    return batch

def gen_tensor_from_pointcloud(pointcloud: np.array) -> torch.FloatTensor:
    return torch.FloatTensor(pointcloud)