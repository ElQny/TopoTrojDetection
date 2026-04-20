import gudhi
import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import pyplot as plt
from plotly import express as px

def plot_persist_diagram(PH):
    gudhi.plot_persistence_diagram(PH)
    plt.show()


def plot_pointcloud(pointcloud: np.array, title: str):
    df = pd.DataFrame(pointcloud, columns = ['x', 'y', 'z'])
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        title = title,
    )
    fig.update_traces(marker=dict(size=2))  # markersize smaller
    fig.show()


def plot_pointcloud_layers(pointcloud:np.array, layers: np.array, layer_names: np.array, title:str):
    df = pd.DataFrame(pointcloud, columns = ['x', 'y', 'z'])
    df['layer_id'] = layers
    df['layer_name'] = layer_names
    df['layer_description'] = df['layer_id'].astype(str) + ': ' + df['layer_name'].astype(str)

    fig = px.scatter_3d(
        df,
        x = 'x',
        y = 'y',
        z = 'z',
        color = 'layer_description',
        title = title,
    )

    fig.update_traces(
        marker=dict(size=2),#markersize smaller
    )
    fig.show()


def plot_pointcloud_trigger(points_clean:np.array, points_backdoor:np.array):
    """
    from: https://github.com/zhenxianglance/PCBA/blob/main/attack_visialization.py
    """
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_clean)
    pcd1.paint_uniform_color(np.array([0.1, 0.1, 0.8]))
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points_backdoor)
    pcd2.paint_uniform_color(np.array([0.8, 0.1, 0.1]))
    # Attack
    o3d.visualization.draw_geometries([pcd1, pcd2])
    # Clean
    o3d.visualization.draw_geometries([pcd1])
