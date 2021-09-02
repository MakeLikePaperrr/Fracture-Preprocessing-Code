import numpy as np
from multiprocessing import freeze_support
from preprocessing import frac_preprocessing


def main():
    # File names and directories:
    output_dir = 'FracData_test_with_inter'
    filename_base = 'FracData_test_with_inter'
    frac_data_raw = np.genfromtxt(filename_base + '.txt')

    # Some constants:
    tolerance_zero = 1e-15
    tolerance_intersect = 1e-15
    decimals = 7
    char_len_vec = np.array([10, 20, 40])
    angle_tol_straighten = 5
    merge_threshold = 0.86  # 0.5, 0.75, 0.86
    angle_tol_remove_segm = np.arctan(0.35) * 180 / np.pi  # (anywhere between 15-25 degrees should be fine)

    reuse_clean = True
    straighten_first = True
    mesh_clean = True
    mesh_raw = True
    num_partition_x = 1
    num_partition_y = 1

    # Reservoir properties:
    height_res = 50
    origin_x = 0
    origin_y = 0
    margin = 25
    length_x = np.max(frac_data_raw[:, [0, 2]])
    length_y = np.max(frac_data_raw[:, [1, 3]])

    box_data = np.array([[origin_x - margin, origin_y - margin],
                         [length_x + margin, origin_y - margin],
                         [length_x + margin, length_y + margin],
                         [origin_x - margin, length_y + margin]])

    frac_preprocessing(frac_data_raw, char_len_vec, output_dir, filename_base, height_res, box_data,
                       angle_tol_straighten, merge_threshold, angle_tol_remove_segm,
                       tolerance_zero, tolerance_intersect, decimals, reuse_clean,
                       straighten_first, mesh_clean, mesh_raw,
                       num_partition_x, num_partition_y)


if __name__ == "__main__":
    freeze_support()
    main()
