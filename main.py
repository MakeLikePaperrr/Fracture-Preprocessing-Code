import numpy as np
from multiprocessing import freeze_support
from preprocessing import frac_preprocessing


def main():
    # File names and directories:
    output_dir = 'FracData_test_with_inter'
    filename_base = 'FracData_test_with_inter'
    frac_data_raw = np.genfromtxt(filename_base + '.txt')

    # Input parameters for cleaning procedure
    char_len_vec = np.array([10, 20, 40])  # vector containing the desired accuracy at which to process the network [m]
    # NOTE: if you do not want to process in a hierarchical sense, scaler (int) input will work too and will process
    # to a single characteristic length

    angle_tol_straighten = 5  # tolerance for straightening fracture segments [degrees]
    merge_threshold = 0.86  # tolerance for merging nodes in algebraic constraint, values on interval [0.5, 0.86] [-]
    angle_tol_remove_segm = np.arctan(0.35) * 180 / np.pi   # tolerance for removing accute intersections, values on interval [15, 25] [degrees]
    decimals = 7  # in order to remove duplicates we need to have fixed number of decimals
    reuse_clean = True  # if True will re-use cleaned network as starting point for hierarchical cleaning
    straighten_first = True  # if True will straighten fractures first before removing accute angles
    mesh_clean = True  # need gmsh installed and callable from command line in order to mesh!!!
    mesh_raw = True  # need gmsh installed and callable from command line in order to mesh!!!
    num_partition_x = 1  # number of partitions for parallel implementation of intersection finding algorithm
    num_partition_y = 1  # " ... "

    # Reservoir properties:
    height_res = 50
    origin_x = 0
    origin_y = 0
    margin = 25
    length_x = np.max(frac_data_raw[:, [0, 2]])
    length_y = np.max(frac_data_raw[:, [1, 3]])

    # Boxdata is only used in meshing and generation of geo file
    box_data = np.array([[origin_x - margin, origin_y - margin],
                         [length_x + margin, origin_y - margin],
                         [length_x + margin, length_y + margin],
                         [origin_x - margin, length_y + margin]])

    frac_preprocessing(frac_data_raw=frac_data_raw, char_len_vec=char_len_vec, output_dir=output_dir,
                       filename_base=filename_base, height_res=height_res, box_data=box_data,
                       angle_tol_straighten=angle_tol_straighten, merge_threshold=merge_threshold,
                       angle_tol_remove_segm=angle_tol_remove_segm, decimals=decimals, reuse_clean=reuse_clean,
                       straighten_first=straighten_first, mesh_clean=mesh_clean, mesh_raw=mesh_raw,
                       num_partition_x=num_partition_x, num_partition_y=num_partition_y)


if __name__ == "__main__":
    freeze_support()
    main()
