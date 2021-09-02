"""
MIT License

Copyright (c) 2021 Stephan de Hoop 		S.dehoop-1@tudelft.nl
                   Denis Voskov 		D.V.Voskov@tudelft.nl
                   Delft University of Technology, the Netherlands

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from cleaning_methods import *
import numpy as np
from calc_intersections_segm_parallel import calc_intersections_segm_parallel
import os


def frac_preprocessing(frac_data_raw, char_len_vec, output_dir, filename_base, height_res, box_data,
                       angle_tol_straighten=7.5, merge_threshold=0.86, angle_tol_remove_segm=15,
                       tolerance_zero=1e-15, tolerance_intersect=1e-15, decimals=7, reuse_clean=True,
                       straighten_first=True, mesh_clean=True, mesh_raw=True,
                       num_partition_x=1, num_partition_y=1):
    print('--------------------------------------')
    print('START preprocessing fracture network')
    tot_partitions = num_partition_x * num_partition_y

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frac_data_raw = np.round(frac_data_raw * 10 ** decimals) * 10 ** (-decimals)
    len_raw_sys = np.sqrt((frac_data_raw[:, 0] - frac_data_raw[:, 2]) ** 2 +
                          (frac_data_raw[:, 1] - frac_data_raw[:, 3]) ** 2)
    print('Remove segments of zero length')
    frac_data_raw = frac_data_raw[np.where(len_raw_sys > tolerance_zero)[0]]
    frac_data_raw = extract_unique_nodes(frac_data_raw)

    print('Number of fracture segments: {:}'.format(frac_data_raw.shape[0]))
    print('Min fracture segment length: {:}'.format(np.min(len_raw_sys)))
    print('Max fracture segment length: {:}'.format(np.max(len_raw_sys)))
    print('Mean fracture segment length: {:}'.format(np.mean(len_raw_sys)))
    print('Cleaning length(s): {:}\n'.format(char_len_vec))

    print('START calculating initial intersections raw input fracture network')
    print('\tNOTE: unoptimized!, can take long for very large networks')
    # First find all intersections:
    system_out_par, frac_order_vec_par, partition_lines = \
        calc_intersections_segm_parallel(frac_data_raw, np.zeros_like(frac_data_raw[:, 0]), tolerance_intersect, tolerance_zero,
                                         num_partition_x, num_partition_y)

    # Stack output from all domain together:
    act_frac_sys = system_out_par[0]
    for ii in range(1, tot_partitions):
        act_frac_sys = np.vstack((act_frac_sys, system_out_par[ii]))

    print('DONE calculating initial intersections raw input fracture network')
    if act_frac_sys.shape != frac_data_raw.shape:
        num_intersections = int((act_frac_sys.shape[0] - frac_data_raw.shape[0])/2)
        print('\tFound {:} intersections in raw input fracture network\n'.format(num_intersections))
    else:
        print('\tNo intersections found in raw input fracture network\n')

    print('Remove duplicated segments\n')
    act_frac_sys = extract_unique_nodes(act_frac_sys)
    act_frac_sys_cln = act_frac_sys
    act_frac_sys_raw = np.array(act_frac_sys, copy=True)

    # Perform loop for all characteristic lengths:
    if type(char_len_vec) is int:
        char_len_vec = [char_len_vec]
    if type(char_len_vec) == np.ndarray:
        if char_len_vec.size == 1:
            char_len_vec = [int(char_len_vec)]

    for char_len in char_len_vec:
        print('START main cleaning loop for l_f={:}'.format(char_len))
        print('\tNOTE: unoptimized!, can take long for very large networks or very small l_f')

        # Start from old network or from previously cleaned:
        if reuse_clean:
            act_frac_sys = act_frac_sys_cln

        act_frac_sys_cln, unique_nodes_cln, incidence_matrix_cln = \
            full_cleaning(act_frac_sys, char_len, decimals, tolerance_zero, angle_tol_straighten, angle_tol_remove_segm,
                          straighten_first, merge_threshold)

        print('DONE main cleaning loop for l_f={:}\n'.format(char_len))

        print('START calculating intersections clean fracture network for coherent mesh')
        print('\tNOTE: unoptimized!, can take long for very large networks')
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
        print('DONE calculating intersections clean fracture network for coherent mesh\n')

        print('START writing fracture network and geo-file for cleaned network')
        # Find unique nodes:
        unique_nodes_cln = np.unique(np.vstack((act_frac_sys_cln[:, :2], act_frac_sys_cln[:, 2:])), axis=0)
        incidence_matrix_cln, order_discr_nodes_cln = create_graph(act_frac_sys_cln, unique_nodes_cln,
                                                                   np.arange(act_frac_sys_cln.shape[0]), tolerance_zero)

        # Write final result to file (frac_sys, geo-file, and mesh final results as well):
        char_len_boundary = char_len * 1

        # Write fracture system to .txt for later use:
        filename_clean = os.path.join(output_dir, filename_base + '_mergefac_' + str(merge_threshold) + '_clean_lc_' + str(char_len) + '.txt')
        write_new_frac_sys(act_frac_sys_cln, filename_clean)

        # Filenames for meshing:
        filename_geo_cln = os.path.join(output_dir, filename_base + '_mergefac_' + str(merge_threshold) + '_clean_lc_' + str(char_len) + '.geo')
        filename_out_cln = os.path.join(output_dir, filename_base + '_mergefac_' + str(merge_threshold) + '_clean_lc_' + str(char_len) + '.msh')
        filename_geo_raw = os.path.join(output_dir, filename_base + '_raw_lc_' + str(char_len) + '.geo')
        filename_out_raw = os.path.join(output_dir, filename_base + '_raw_lc_' + str(char_len) + '.msh')

        # Create geo-file and mesh result (clean):
        create_geo_file(act_frac_sys_cln, unique_nodes_cln, incidence_matrix_cln, filename_geo_cln, decimals,
                        height_res, char_len, box_data, char_len_boundary)
        print('DONE writing fracture network and geo-file for cleaned network\n')

        if mesh_clean:
            print('START meshing cleaned network')
            print('\tNOTE: In gmsh you need to have under Options -> Geometry -> General -> uncheck "Remove duplicate ..." otherwise meshing will crash/take too long')
            mesh_geo_file(filename_geo_cln, filename_out_cln)
            print('DONE meshing cleaned network\n')

        print('START writing geo-file for raw network')
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
        print('DONE writing geo-file for raw network\n')

        if mesh_raw:
            print('START meshing raw network')
            print('\tNOTE: unoptimized!, can take long for very large raw networks')
            print('\tNOTE: In gmsh you need to have under Options -> Geometry -> General -> uncheck "Remove duplicate ..." otherwise meshing will crash/take too long')
            mesh_geo_file(filename_geo_raw, filename_out_raw)
            print('DONE meshing raw network\n')

    print('Preprocessing succesfully finished')
    print('-----------------------------------')
    return 0
