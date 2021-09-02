from cleaning_methods import *
import numpy as np
from calc_intersections_segm_parallel import calc_intersections_segm_parallel
from multiprocessing import Process, freeze_support
import os


def main():
    # Some constants:
    tolerance_zero = 1e-15
    tolerance_intersect = 1e-15
    decimals = 7
    inf_value = np.inf
    # char_len_vec = np.array([4, 8, 16, 32, 64, 128])
    char_len_vec = np.array([10, 20, 40])
    angle_tol_straighten = 5
    merge_threshold = 0.86  # 0.5, 0.75, 0.86
    angle_tol_remove_segm = np.arctan(0.35) * 180 / np.pi  # (anywhere between 15-25 degrees should be fine)

    # angle_tol_straighten = 0
    # angle_tol_remove_segm = 0

    # angle_tol_remove_segm = 45

    reuse_clean = True
    straighten_first = True
    mesh_raw = True
    num_partition_x = 1
    num_partition_y = 1
    tot_partitions = num_partition_x * num_partition_y

    # Filename for input array:
    # filename_base = 'Whitby_norm'
    # filename_base = 'Brejoes_norm_with_inter'  # Brejoes_fracdata_norm_with_intersections
    filename_base = 'FracData_test_with_inter'
    filename_raw = filename_base + '.txt'

    if not os.path.exists(filename_base):
        os.makedirs(filename_base)

    # Load fracture data set:
    act_frac_sys_raw = np.genfromtxt(filename_raw)
    act_frac_sys_raw = np.round(act_frac_sys_raw * 10 ** decimals) * 10 ** (-decimals)
    len_raw_sys = np.sqrt((act_frac_sys_raw[:, 0] - act_frac_sys_raw[:, 2]) ** 2 +
                          (act_frac_sys_raw[:, 1] - act_frac_sys_raw[:, 3]) ** 2)
    act_frac_sys_raw = act_frac_sys_raw[np.where(len_raw_sys > tolerance_zero)[0]]
    act_frac_sys_raw = extract_unique_nodes(act_frac_sys_raw)

    # Reservoir properties:
    height_res = 50
    origin_x = 0
    origin_y = 0
    margin = 25
    length_x = np.max(act_frac_sys_raw[:, [0, 2]])
    length_y = np.max(act_frac_sys_raw[:, [1, 3]])

    box_data = np.array([[origin_x - margin, origin_y - margin],
                         [length_x + margin, origin_y - margin],
                         [length_x + margin, length_y + margin],
                         [origin_x - margin, length_y + margin]])

    # First find all intersections:
    system_out_par, frac_order_vec_par, partition_lines = \
        calc_intersections_segm_parallel(act_frac_sys_raw, np.zeros_like(act_frac_sys_raw[:, 0]), tolerance_intersect, tolerance_zero,
                                         num_partition_x, num_partition_y)

    # Stack output from all domain together:
    act_frac_sys = system_out_par[0]
    for ii in range(1, tot_partitions):
        act_frac_sys = np.vstack((act_frac_sys, system_out_par[ii]))

    act_frac_sys = extract_unique_nodes(act_frac_sys)
    #
    # act_frac_sys = np.array(act_frac_sys_raw, copy=True)
    # act_frac_sys_raw_int = np.array(act_frac_sys, copy=False)

    # Perform loop for all characteristic lengths:
    for jj, char_len in enumerate(char_len_vec):
        # Start from old network or from previously cleaned:
        if reuse_clean and jj > 0:
            act_frac_sys = act_frac_sys_cln

        act_frac_sys_cln, unique_nodes_cln, incidence_matrix_cln = \
            full_cleaning(act_frac_sys, char_len, decimals, tolerance_zero, angle_tol_straighten, angle_tol_remove_segm,
                          straighten_first, merge_threshold)

        # Make sure to catch all the intersections after procedure for coherent mesh:
        system_out_par, frac_order_vec_par, partition_lines = \
            calc_intersections_segm_parallel(act_frac_sys_cln, np.zeros_like(act_frac_sys_cln[:, 0]),
                                             tolerance_intersect, tolerance_zero,
                                             num_partition_x, num_partition_y)

        # Stack output from all domain together:
        act_frac_sys_cln = system_out_par[0]
        for ii in range(1, tot_partitions):
            act_frac_sys_cln = np.vstack((act_frac_sys_cln, system_out_par[ii]))

        act_frac_sys_cln = extract_unique_nodes(act_frac_sys_cln)

        # Find unique nodes:
        unique_nodes_cln = np.unique(np.vstack((act_frac_sys_cln[:, :2], act_frac_sys_cln[:, 2:])), axis=0)
        incidence_matrix_cln, order_discr_nodes_cln = create_graph(act_frac_sys_cln, unique_nodes_cln,
                                                                   np.arange(act_frac_sys_cln.shape[0]), tolerance_zero)

        # Write final result to file (frac_sys, geo-file, and mesh final results as well):
        char_len_boundary = char_len * 1

        # Write fracture system to .txt for later use:
        filename_clean = os.path.join(filename_base, filename_base + '_mergefac_' + str(merge_threshold) + '_clean_lc_' + str(char_len) + '.txt')
        write_new_frac_sys(act_frac_sys_cln, filename_clean)

        # Filenames for meshing:
        filename_geo_cln = os.path.join(filename_base, filename_base + '_mergefac_' + str(merge_threshold) + '_clean_lc_' + str(char_len) + '.geo')
        filename_out_cln = os.path.join(filename_base, filename_base + '_mergefac_' + str(merge_threshold) + '_clean_lc_' + str(char_len) + '.msh')
        filename_geo_raw = os.path.join(filename_base, filename_base + '_raw_lc_' + str(char_len) + '.geo')
        filename_out_raw = os.path.join(filename_base, filename_base + '_raw_lc_' + str(char_len) + '.msh')

        # Create geo-file and mesh result (clean):
        create_geo_file(act_frac_sys_cln, unique_nodes_cln, incidence_matrix_cln, filename_geo_cln, decimals,
                        height_res, char_len, box_data, char_len_boundary)

        mesh_geo_file(filename_geo_cln, filename_out_cln)

        if mesh_raw:
            # Create geo-file and mesh result (raw):
            order_discr_org = np.arange(act_frac_sys_raw.shape[0])
            unique_nodes_org = np.unique(np.vstack((act_frac_sys_raw[:, :2], act_frac_sys_raw[:, 2:])), axis=0)
            incidence_matrix_org, order_discr_nodes_org = create_graph(act_frac_sys_raw, unique_nodes_org, order_discr_org,
                                                                   tolerance_zero)

            unique_segm_list, segm_count = np.unique(incidence_matrix_org[:, 1], return_counts=True)
            mask_2 = segm_count != 2

            if any(mask_2):
                print('SEGMENT WITH MORE THAN 2 NODES, PROBABLY TO LOOSE ZERO FOR FINDING SIMILAR NODES!!!')
                sys.exit(0)

            create_geo_file(act_frac_sys_raw, unique_nodes_org, incidence_matrix_org, filename_geo_raw, decimals,
                            height_res, char_len, box_data, char_len_boundary)

            mesh_geo_file(filename_geo_raw, filename_out_raw)


if __name__ == "__main__":
    freeze_support()
    main()
