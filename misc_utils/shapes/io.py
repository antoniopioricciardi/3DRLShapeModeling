import robust_laplacian as rl
import open3d as o3d
import numpy as np
import scipy as spshape_name
import os
from misc_utils.shapes.operations import *

def simplify_mesh(mesh_in):
    # mesh_in = o3dtut.get_bunny_mesh()
    print(
        f'Input mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles'
    )
    # o3d.visualization.draw_geometries([mesh_in])

    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 28
    print(f'voxel_size = {voxel_size:e}')
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    print(
        f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
    )
    # o3d.visualization.draw_geometries([mesh_smp])

    # voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 16
    print(f'voxel_size = {voxel_size:e}')
    # mesh_smp = mesh_in.simplify_vertex_clustering(
    #     voxel_size=voxel_size,
    #     contraction=o3d.geometry.SimplificationContraction.Average)
    # print(
    #     f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
    # )
    # o3d.visualization.draw_geometries([mesh_smp])
    return mesh_smp

def generate_new_mesh_from_file(new_vertices, source_shape_path='./shapes/25.ply', simplify=False):
    Mesh = o3d.io.read_triangle_mesh(source_shape_path)
    if simplify:
        Mesh = simplify_mesh(Mesh)
    F = np.asarray(Mesh.triangles)
    Mesh.compute_vertex_normals()
    # o3d.visualization.draw([Mesh])
    Mesh_new = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(new_vertices),
                                         triangles=o3d.utility.Vector3iVector(F))
    return Mesh_new


def load_mesh(shape_path):
    return o3d.io.read_triangle_mesh(shape_path)


def save_mesh(mesh, path):
    o3d.io.write_triangle_mesh(path, mesh)


def generate_new_mesh(v, f):
    # o3d.visualization.draw([Mesh])
    Mesh_new = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(v),
                                         triangles=o3d.utility.Vector3iVector(f))
    return Mesh_new


def create_sphere_from_mesh(V, F):
    v = mean_curvature_flow(V, F)
    mesh = generate_new_mesh(v, F)
    mesh.compute_vertex_normals()

    # o3d.io.write_triangle_mesh('sphere.ply', mesh)
    return mesh, v, F

# load_mesh(shape_path='./shapes/tr_reg_000.ply', simplify=False)

# load_mesh('shapes/tr_reg_017.ply')

# Mesh_new = o3d.geometry.TriangleMesh(vertices = o3d.utility.Vector3dVector(U),
# triangles = o3d.utility.Vector3iVector(F))
# o3d.visualization.draw([Mesh_new, Mesh])
#
# Mesh_new.compute_vertex_normals()
# Mesh.compute_vertex_normals()
# o3d.visualization.draw([Mesh_new, Mesh])