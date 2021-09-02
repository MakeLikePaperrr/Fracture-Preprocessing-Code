"""
This file describes the function calc_intersections_segm, which is called in the main cleaning function. This
function finds the intersections of the fractures in act_frac_sys, and splits these fractures at the intersection
points. The list act_frac_sys is then returned with the updated fractures.

Author: Ole de Koning

Last updated: 12/12/2020 by Ole de Koning
"""

import numpy as np
import time
from find_parametric_intersect import find_parametric_intersect


def calc_intersections_segm(act_frac_sys,frac_order_vec, tolerance_intersect, tolerance_zero):
    """
    :param act_frac_sys: A list where all rows describe a fracture in the system, and the four columns
                         describe the following: Column 0 = X value of the start of a fracture
                                                 Column 1 = Y value of the start of a fracture
                                                 Column 2 = X value of the end of a fracture
                                                 Column 3 = Y value of the end of a fracture

    :param frac_set_vec: This is a list where groups of fractures are defined that are close to each other.

    :param frac_order_vec: Array of order identifiers of the fractures, determined in change_order_discr.
    :param tolerance_intersect: A value that was set to determine whether two fractures intersect of not.
                                If t and s, the parametric distances along fracture 1 and 2 respectively are
                                smaller than this tolerance, it is counted as an intersection.

    :param tolerance_zero: This tolerance is used to check whether a segment is non-zero. In this case, the
                           length of the segment must be larger than this value.

    :return: act_frac_sys: This returned matrix is similar to the input act_frac_sys, except that it now takes
                           all intersection into account.

    :return: frac_set_vec: This returned matrix is similar to the input frac_set_vec, except that it now takes
                           all intersection into account. The groups formed in this list will thus differ from the
                           ones in the input.
    """

    # Allocate large array for new points:
    n_fracs = act_frac_sys.shape[0]
    max_new_pts = 10000
    max_length_new_segm = n_fracs + max_new_pts * 2

    # Segment list global:
    new_frac_order_vec = np.zeros(max_length_new_segm*2)
    new_points = np.zeros((max_new_pts, 2))
    ith_jj = np.zeros(max_new_pts)
    new_fract_sys = np.zeros((max_length_new_segm, 4))

    ith_pt = -1
    glob_segm_count = 0

    for ii in range(0, n_fracs):

        # Obtaining the x and y coords of the start and end of the frac
        ith_old = ith_pt + 1
        ii_frac = act_frac_sys[ii, :]

        for jj in range(ii+1, n_fracs):
            jj_frac = act_frac_sys[jj, :]

            if np.min(jj_frac[[0, 2]]) > np.max(ii_frac[[0, 2]]) or np.max(jj_frac[[0, 2]]) < np.min(ii_frac[[0, 2]]):
                continue
            elif np.min(jj_frac[[1, 3]]) > np.max(ii_frac[[1, 3]]) or np.max(jj_frac[[1, 3]]) < np.min(ii_frac[[1, 3]]):
                continue

            # Only store intersections of segments that don't already share a node:
            if not (np.linalg.norm(ii_frac[:2] - jj_frac[:2]) < tolerance_intersect or
                    np.linalg.norm(ii_frac[:2] - jj_frac[2:]) < tolerance_intersect or
                    np.linalg.norm(ii_frac[2:] - jj_frac[:2]) < tolerance_intersect or
                    np.linalg.norm(ii_frac[2:] - jj_frac[2:]) < tolerance_intersect):

                t, s, int_coord = find_parametric_intersect(ii_frac, jj_frac)

                if (t >= (0 - tolerance_intersect) and t <= (1 + tolerance_intersect)) and \
                        (s >= (0 - tolerance_intersect) and s <= (1 + tolerance_intersect)):

                    ith_pt = ith_pt + 1
                    new_points[ith_pt, :] = int_coord
                    ith_jj[ith_pt] = jj

        if ii != 0:
            prev_jj_int = new_points[np.where(ith_jj == ii), :][0]
        else:
            prev_jj_int = np.zeros((0, 2))

        new_ii_int = new_points[ith_old:ith_pt+1, :]

        num_prev_int = prev_jj_int.shape[0]
        num_new_int = new_ii_int.shape[0]

        # Check if there even are fractures. If not, we go on to the next fracture ii.
        if num_prev_int == 0 and num_new_int == 0:
            glob_segm_count += 1
            new_fract_sys[glob_segm_count:(glob_segm_count + 1), :] = ii_frac
            new_frac_order_vec[glob_segm_count: (glob_segm_count + 1)] = frac_order_vec[ii]
            glob_segm_count += 1
            continue

        tot_new_pts = 2 + num_new_int + num_prev_int

        tot_loc_pts_list = np.zeros((tot_new_pts, 2))
        tot_loc_pts_list[0, :] = act_frac_sys[ii, :2]
        tot_loc_pts_list[-1, :] = act_frac_sys[ii, 2:]
        tot_loc_pts_list[1:num_prev_int+1, :] = prev_jj_int
        tot_loc_pts_list[num_prev_int+1:num_new_int+num_prev_int+1, :] = new_ii_int

        tot_loc_pts_list = tot_loc_pts_list[np.lexsort((tot_loc_pts_list[:, 1], tot_loc_pts_list[:, 0]))]

        tot_new_segm = tot_loc_pts_list.shape[0] - 1
        # Now, we have an array with all x and y value of the points where the fracture ii should be split at.
        tot_loc_segm_list = np.zeros((tot_new_segm, 4))
        for mm in range(0, tot_new_segm):
            tot_loc_segm_list[mm, :] = [tot_loc_pts_list[mm, 0],
                                        tot_loc_pts_list[mm, 1],
                                        tot_loc_pts_list[mm + 1, 0],
                                        tot_loc_pts_list[mm + 1, 1]]

        new_fract_sys[glob_segm_count:(glob_segm_count + tot_new_segm), :] = tot_loc_segm_list

        new_frac_order_vec[glob_segm_count: (glob_segm_count + tot_new_segm)] = frac_order_vec[ii]

        glob_segm_count += tot_new_segm

    act_frac_sys = new_fract_sys[:glob_segm_count, :]
    new_frac_order_vec = new_frac_order_vec[:glob_segm_count]

    # Determine length of new "main" segments:
    len_segm_new = np.sqrt((act_frac_sys[:, 0] - act_frac_sys[:, 2])*(act_frac_sys[:, 0] - act_frac_sys[:, 2]) +
                           (act_frac_sys[:, 1] - act_frac_sys[:, 3])*(act_frac_sys[:, 1] - act_frac_sys[:, 3]))

    # Remove non-zero fracture segments: This comment does not make sense, probably meant the opposite
    nonzero_segm = np.where(len_segm_new > tolerance_zero)[0]
    act_frac_sys = act_frac_sys[nonzero_segm, :]

    frac_order_vec = new_frac_order_vec[nonzero_segm]

    return act_frac_sys, frac_order_vec
