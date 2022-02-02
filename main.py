import numpy as np
from multiprocessing import freeze_support
from preprocessing_code import frac_preprocessing
import os


def main():
    # File names and directories:
    output_dir = 'output_Whitby'
    filename_base = 'Whitby'
    frac_data_raw = np.genfromtxt(os.path.join('Datasets', 'Whitby', 'Whitby_with_intersections.txt'))
    # frac_data_raw = np.genfromtxt(os.path.join('Datasets', 'Whitby', 'Whitby_raw.txt'))

    # Input parameters for cleaning procedure
    char_len = 16  # characteristic length for cleaning and mesh generation [m]
    angle_tol_straighten = 7.5  # tolerance for straightening fracture segments [degrees]
    merge_threshold = 0.86  # tolerance for merging nodes in algebraic constraint, values on interval [0.5, 0.86] [-]
    angle_tol_remove_segm = np.arctan(0.35) * 180 / np.pi   # tolerance for removing accute intersections, values on interval [15, 25] [degrees]
    decimals = 7  # in order to remove duplicates we need to have fixed number of decimals
    mesh_clean = False  # need gmsh installed and callable from command line in order to mesh!!!
    mesh_raw = False  # need gmsh installed and callable from command line in order to mesh!!!
    num_partition_x = 4  # number of partitions for parallel implementation of intersection finding algorithm
    num_partition_y = 4  # " ... "

    frac_preprocessing(frac_data_raw, char_len, output_dir=output_dir, filename_base=filename_base, merge_threshold=merge_threshold,
                       height_res=50, angle_tol_small_intersect=angle_tol_remove_segm, apertures_raw=None, box_data=None, margin=25,
                       mesh_clean=mesh_clean, mesh_raw=mesh_raw, angle_tol_straighten=angle_tol_straighten, straighten_after_cln=True, decimals=decimals,
                       tolerance_zero=1e-10, tolerance_intersect=1e-10, calc_intersections_before=False, calc_intersections_after=False,
                       num_partition_x=num_partition_x, num_partition_y=num_partition_y, partition_fractures_in_segms=True, matrix_perm=1, correct_aperture=False,
                       small_angle_iter=2, char_len_mult=1, char_len_boundary=None, main_algo_iters=1)


if __name__ == "__main__":
    freeze_support()
    main()
