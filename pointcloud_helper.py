import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_pointcloud(pointcloud: np.array, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]

    ax.scatter(x, y, z)
    ax.set_title(title)
    plt.show()

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
    """
        Generates a random point on the unit sphere
    """
    while True:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform(-1, 1)
        point = np.array([x, y, z], dtype=np.float32)
        if is_in_unit_sphere(point):
            return point

def create_sample_pointcloud(N: int) -> np.array:
    """
        Creates a sample pointcloud with N points in it
    """
    print("Creating sample pointcloud")
    pointcloud = []
    for n in range(N):
        pointcloud.append(random_point_in_unit_sphere())
    pointcloud = np.array(pointcloud, dtype=np.float32)
    return pointcloud

def is_in_cube(point: np.array, cube: np.array, granularity: int) -> bool: #cube: x_end, y_end, z_end
    """
        Checks if a point is inside the cube
        Cube is an array of x_end, y_end, z_end, if subtracting the stepsize,
        one gets x_start, y_start, z_start
    """
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
    """
        Generates a structure made up from cubes within a given
        granularity
    """
    print("Generating cubes")
    stepsize = calc_stepsize(granularity)
    cube_list = []
    for x in np.arange(-1, 1, stepsize): #excluding upper range
        for y in np.arange(-1, 1, stepsize):
            for z in np.arange (-1, 1, stepsize):
                cube_list.append([x+stepsize,y+stepsize,z+stepsize]) #include upper range by only saving upper range
    return cube_list

def choose_sub_pointclouds(pointcloud: np.array, granularity: int) -> list:
    """
        separates a pointcloud into cubes by a given granularity
    """
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
    """
        perturbs the points of a pointcloud by adding a random number between
        -max_perturbation and +max_perturbation
    """
    x_temp = point[0] + random.uniform(-max_perturbation, max_perturbation)
    y_temp = point[1] + random.uniform(-max_perturbation, max_perturbation)
    z_temp = point[2] + random.uniform(-max_perturbation, max_perturbation)
    temp_point = np.array([x_temp, y_temp, z_temp])
    return temp_point

def perturb_points_in_cube(
        pointcloud: np.array,
        points_in_subcube: list,
        cube: np.array,
        granularity:int,
        decimal_positions: int = None #for rounding the points
) -> np.array: #returns pointcloud with perturbed points
    """
        perturbs the points in a given cube, possibly with rounding as a light discretization
    """

    # return pointcloud.copy() #comment in for testing and uncomment block below
    perturbed_pointcloud = pointcloud.copy()
    max_perturbation = calc_stepsize(granularity) / 10

    for point_index in points_in_subcube:
        point = pointcloud[point_index]

        for _ in range(10): #maximum of 10 tries for each point to be within box-bounds after perturbation
            perturbed_point = perturb_point(point, max_perturbation)

            if decimal_positions is not None: #light discretization with rounding
                perturbed_point = round_point(perturbed_point, decimal_positions)

            if is_in_cube(perturbed_point, cube, granularity) and is_in_unit_sphere(perturbed_point):
                perturbed_pointcloud[point_index] = perturbed_point
                break
    return perturbed_pointcloud

def transpose_and_batch_pointclouds_to_tensor(pointclouds: np.array) -> torch.FloatTensor:
    #(B, N, 3)->(B, 3, N)
    batch = np.transpose(pointclouds, (0, 2, 1))
    return torch.FloatTensor(batch.astype(np.float32)) #FloatTensor expects float32

def calc_stepsize(granularity: int) -> float:
    """
        calculates the stepsize for a given granularity
        stepsize = 2/granularity because edges go from -1 to 1
        so total length = 2 (unit sphere)
    """
    return 2 / granularity

def generate_perturbed_pointcloud_batch(batch_size, c_idx: int, cubes, device, example_pointcloud, granularity, points_in_cube, round_decimals = None) -> torch.Tensor:
    print("Generating perturbed pointcloud batch")
    perturbed_pointclouds = []
    for b in range(batch_size):
        temp_perturbed_pc = perturb_points_in_cube(
            pointcloud=example_pointcloud,
            points_in_subcube=points_in_cube,
            cube=cubes[c_idx],
            granularity=granularity,
            decimal_positions=round_decimals
        )
        perturbed_pointclouds.append(temp_perturbed_pc)

    perturbed_pointclouds = np.array(perturbed_pointclouds)
    tensor = transpose_and_batch_pointclouds_to_tensor(perturbed_pointclouds).to(device)
    return tensor


# Functions for generating a small Sphere with Radius r and moving it / resizing it
def possible_sphere_centers(step: float):
    centers = []
    coordinates = np.arange(-1.0, 1.0 + 1e-10, step) #adding e-10 so the upper border is included
    for x in coordinates:
        for y in coordinates:
            for z in coordinates:
                centers.append(np.array([x, y, z]))
    return centers

def generate_pcba_sphere_from_center(center:np.array, radius:float, npoints:int) -> np.array:
    #primary source for this function: https://github.com/zhenxianglance/PCBA/blob/main/attack_utils.py
    sphere_points = np.zeros([npoints, 3])

    for n in range(npoints):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        sphere_points[n,0] = radius * np.sin(theta) * np.cos(phi) + center[0] #x value
        sphere_points[n,1] = radius * np.sin(theta) * np.sin(phi) + center[1] #y value
        sphere_points[n,2] = radius * np.cos(theta) + center[2] #z value
    return sphere_points

def generate_spheres_from_center(step: float, radius, npoints:int):
    pointclouds = []
    # information = []

    centers=possible_sphere_centers(step)

    for center in centers: #generates a sphere for each possible center
        pointcloud = generate_pcba_sphere_from_center(center, radius, npoints)
        pointclouds.append(pointcloud)
        # information.append({
        #     "radius": radius,
        #     "center": center
        # })
    pointclouds = np.array(pointclouds)
    # return pointclouds, information
    return pointclouds

def generate_radius_batch(
        clean_pointcloud: np.array,
        center: np.array,
        radius_min: float,
        radius_max: float,
        radius_step: float,
        number_of_points_trigger: int
        ):
    # res = r_stepsize between r_min & r_max!
    pointclouds = []
    radii = np.arange(radius_min, radius_max + 1e-10, radius_step) #adding +1e-10 so the upper border is included

    for radius in radii:
        trigger = generate_pcba_sphere_from_center(center, radius, number_of_points_trigger)
        overlay = overlay_trigger_on_pointcloud(clean_pointcloud, trigger)
        pointclouds.append(overlay)

    pointclouds = np.array(pointclouds)
    tensor = transpose_and_batch_pointclouds_to_tensor(pointclouds) # (radiuscount, pointcount, 3) -> (radiuscount, 3, pointcount)
    return tensor, radii


def overlay_trigger_on_pointcloud(clean_pointcloud:np.array, trigger:np.array) -> np.array:
    return np.concatenate([clean_pointcloud, trigger], axis=0)

def load_off_file(filepath: str) -> np.array:
    with open(filepath, 'r') as file:
        first_line = file.readline()

        if first_line.strip() == 'OFF':
            header = file.readline().strip()
        else:
            raise ValueError(f'File {filepath} is not a valid OFF file.')

        headers = header.split(' ')
        cnt_vertices = int(headers[0])

        vertices = []
        for vert_idx in range(cnt_vertices):
            line = file.readline().strip()
            if not line:
                continue
            coordinates = line.split(' ') #separates x,y,z values to strings!
            x = float(coordinates[0])
            y = float(coordinates[1])
            z = float(coordinates[2]) #strings to float
            vertices.append([x,y,z]) # appends points, might overshoot possible points for NN?

    return np.array(vertices)

# test: discretizing the pointclouds:
def round_point(point: np.array, decimal_positions: int) -> np.array:
    return np.round(point, decimals=decimal_positions)