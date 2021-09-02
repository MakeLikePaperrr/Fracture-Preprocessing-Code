"""
Function that merges the domain partitions
"""

import numpy as np

def merge_domain(act_frac_sys_list, frac_set_vec, partition_lines, number_partitions_x):
    """

    :param act_frac_sys_list: Array of fractures that were once partitioned, and must now be merged back together.
    :param frac_set_vec_list: Array of grouping identifiers.
    :param frac_order_vec_list: Array of priority identifiers.
    :param partition_lines: Lines over which the domain was partitioned at the start of parallelizing.
    :param number_partitions_x: The numbers of partitions in x-direction.
    :return:
    """

    act_frac_sys = np.vstack(act_frac_sys_list)[:, :4]
    uniq_inds = np.unique(act_frac_sys, return_index=True, axis=0)[1]
    act_frac_sys = act_frac_sys[uniq_inds]

    merge_tolerance = 1e-5

    for i in range(partition_lines.shape[0]):
        if i < number_partitions_x - 1:
            direction = 0
            other_dir = 1
        else:
            direction = 1
            other_dir = 0

        partition_line = partition_lines[i]

        # Each line will have one constant value, either on the x- or y-axis.
        const_value_par_line = partition_line[direction]

        inds_left = np.where(np.abs(act_frac_sys[:, direction + 2] - const_value_par_line) <= merge_tolerance)
        inds_right = np.where(np.abs(act_frac_sys[:, direction] - const_value_par_line) <= merge_tolerance)
        inds_afs = np.hstack((inds_left, inds_right))

        segments_left = act_frac_sys[inds_left]
        segments_right = act_frac_sys[inds_right]

        segments_left_slope = np.abs((segments_left[:, 1] - segments_left[:, 3]) / (segments_left[:, 0] - segments_left[:, 2]))
        segments_right_slope = np.abs((segments_right[:, 1] - segments_right[:, 3]) / (segments_right[:, 0] - segments_right[:, 2]))
        # Delete intersection information, it is not needed and stands in the way of merging segments.
        segments_left = np.hstack((segments_left, segments_left_slope.reshape((len(segments_left_slope), 1))))
        segments_right = np.hstack((segments_right, segments_right_slope.reshape((len(segments_right_slope), 1))))

        # Grouping the left and right part or bottom and top part of the newly merged fracture together.
        if segments_left.shape[0] >= segments_right.shape[0]:
            restored_fracs = np.zeros((segments_left.shape[0], 4))
            i = 0
            for segm in segments_left:
                inds_val = np.where(segments_right[:, other_dir] == segm[other_dir + 2])[0]
                inds_slope = np.where(np.abs(segments_right[:, 4] - segm[4] < merge_tolerance))[0]
                inds_restore = np.intersect1d(inds_val, inds_slope)
                if len(inds_restore) == 1:
                    restored_fracs[i, :2] = segm[:2]
                    restored_fracs[i, 2:] = segments_right[inds_restore, 2:4]
                    i += 1

        else:
            restored_fracs = np.zeros((segments_right.shape[0], 4))
            i = 0
            for segm in segments_right:
                inds_val = np.where(segments_left[:, other_dir + 2] == segm[other_dir])[0]
                inds_slope = np.where(np.abs(segments_left[:, 4] - segm[4] < merge_tolerance))[0]
                inds_restore = np.intersect1d(inds_val, inds_slope)
                if len(inds_restore) == 1:
                    restored_fracs[i, :2] = segments_left[inds_restore, :2]
                    restored_fracs[i, 2:] = segm[2:4]
                    i += 1


        # Deleting the previous fractures that are now merged and adding the fractures to act_frac_sys.
        act_frac_sys = np.delete(act_frac_sys, inds_afs, axis=0)
        act_frac_sys = np.vstack((act_frac_sys, restored_fracs))

        # Update frac_set_vec

    return act_frac_sys, frac_set_vec
