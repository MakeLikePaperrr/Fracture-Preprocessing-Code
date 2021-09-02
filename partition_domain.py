"""
Function that partitions the domain
"""
import numpy as np
import math
from find_parametric_intersect import find_parametric_intersect


def partition_domain(act_frac_sys, frac_order_vec, tolerance_intersect, number_partitions_x, number_partitions_y):
    """

    :param frac_order_vec:
    :param act_frac_sys:
    :param tolerance_intersect:
    :param number_partitions_x:
    :param number_partitions_y:
    :return:
    """

    # Define lowest and highest possible x and y values.
    xmin = act_frac_sys[:, [0,2]].min(axis=None)
    xmax = act_frac_sys[:, [0,2]].max(axis=None)
    ymin = act_frac_sys[:, [1,3]].min(axis=None)
    ymax = act_frac_sys[:, [1,3]].max(axis=None)
    interval_x = (xmax - xmin) / number_partitions_x
    interval_y = (ymax - ymin) / number_partitions_y

    # Assume the maximum values found above define the domain.
    # Then, to partition the domain:
    partitions_x = np.zeros((number_partitions_x - 1, 4))
    for i in range(1, number_partitions_x):
        x_part = xmin + interval_x * i
        partitions_x[i - 1, :] = np.array([x_part, ymin, x_part, ymax])

    partitions_y = np.zeros((number_partitions_y - 1, 4))
    for j in range(1, number_partitions_y):
        y_part = ymin + interval_y * j
        partitions_y[j - 1, :] = np.array([xmin, y_part, xmax, y_part])

    # This array will contain information about the partitioning lines which which can be used to find intersections.
    # [x0, y0, x1, y1]
    partitions = np.vstack((partitions_x, partitions_y))

    # We use this little trick to make sure the subdomains are determined correctly.
    act_frac_sys[np.where(act_frac_sys == xmax)] = xmax - 0.01
    act_frac_sys[np.where(act_frac_sys == ymax)] = ymax - 0.01

    # Variables used to store information in an array later in the program.
    old_index = 0
    new_index = 0

    subdomain_sys = np.transpose([np.floor((act_frac_sys[:, 0] - xmin) / interval_x),
                                  np.floor((act_frac_sys[:, 1] - ymin) / interval_y),
                                  np.floor((act_frac_sys[:, 2] - xmin) / interval_x),
                                  np.floor((act_frac_sys[:, 3] - ymin) / interval_y)])

    # Change back what we did to get the right subdomains
    act_frac_sys[np.where(act_frac_sys == xmax - 0.01)] = xmax
    act_frac_sys[np.where(act_frac_sys == ymax - 0.01)] = ymax

    fracs_to_part_x = np.where(subdomain_sys[:, 0] != subdomain_sys[:, 2])[0]
    fracs_to_part_y = np.where(subdomain_sys[:, 1] != subdomain_sys[:, 3])[0]

    # An array of indices referring to fractures that must be split due to partitioning.
    fracs_to_part = np.union1d(fracs_to_part_x, fracs_to_part_y)
    part_frac_sys = act_frac_sys[fracs_to_part]
    part_frac_subdomains = subdomain_sys[fracs_to_part]

    # CHECK
    tot_new_fracs = np.sum(np.abs(subdomain_sys[fracs_to_part, 2] - subdomain_sys[fracs_to_part, 0]) + \
                           np.abs(subdomain_sys[fracs_to_part, 3] - subdomain_sys[fracs_to_part, 1]), dtype=int) + len(fracs_to_part)

    # Array where all newly found partitioned fractures will be stored. The number of rows is pretty arbitrary.
    part_fracs = np.zeros((tot_new_fracs, 5))

    # Arrays where all information is stored to, in the end, form frac_order_vec_list.
    part_frac_order_vec = np.zeros(tot_new_fracs)

    # To clear some memory, the subdomains which are in part_frac_subdomains can now be deleted from the original array.
    subdomain_sys = np.delete(subdomain_sys, fracs_to_part, axis=0)

    ii = -1

    for ii_frac in part_frac_sys:

        ii += 1

        # The subdomains of points in this fracture
        ii_subdomains = part_frac_subdomains[ii, :]

        # I do not expect a fracture to cross more than 6 partition lines. Still an estimate though.
        num_ints = int(abs(ii_subdomains[2] - ii_subdomains[0]) + abs(ii_subdomains[3] - ii_subdomains[1]))
        part_int = np.zeros((num_ints, 2))

        # Counts the amount of intersections between the given ii fracture and all partitioning lines.
        int_counter = 0

        # Partition IDs. [subdomain xmin, subdomain xmax, subdomain ymin, subdomain ymax]
        # (an offset was added to subdomains of y to establish the difference between x and y)
        partition_ids = [int(min(ii_subdomains[0], ii_subdomains[2])),
                         int(max(ii_subdomains[0], ii_subdomains[2])),
                         int(number_partitions_x - 1 + min(ii_subdomains[1], ii_subdomains[3])),
                         int(number_partitions_x - 1 + max(ii_subdomains[1], ii_subdomains[3]))]

        # x partitions
        for jj_part in partitions[partition_ids[0]:partition_ids[1]]:
            t, s, int_coord = find_parametric_intersect(ii_frac, jj_part)

            if (t >= (0 - tolerance_intersect) and t <= (1 + tolerance_intersect)) and \
               (s >= (0 - tolerance_intersect) and s <= (1 + tolerance_intersect)):

                # Only store intersections of segments that don't already share a node:
                if not (np.linalg.norm(ii_frac[:2] - jj_part[:2]) < tolerance_intersect or
                        np.linalg.norm(ii_frac[:2] - jj_part[2:]) < tolerance_intersect or
                        np.linalg.norm(ii_frac[2:] - jj_part[:2]) < tolerance_intersect or
                        np.linalg.norm(ii_frac[2:] - jj_part[2:]) < tolerance_intersect):

                    # Store the intersection coordinates in part_int
                    part_int[int_counter, :] = np.array([int_coord[0], int_coord[1]])
                    int_counter += 1

        # y partitions
        for jj_part in partitions[partition_ids[2]:partition_ids[3]]:

            t, s, int_coord = find_parametric_intersect(ii_frac, jj_part)

            if (t >= (0 - tolerance_intersect) and t <= (1 + tolerance_intersect)) and \
               (s >= (0 - tolerance_intersect) and s <= (1 + tolerance_intersect)):

                # Only store intersections of segments that don't already share a node:
                if not (np.linalg.norm(ii_frac[:2] - jj_part[:2]) < tolerance_intersect or
                        np.linalg.norm(ii_frac[:2] - jj_part[2:]) < tolerance_intersect or
                        np.linalg.norm(ii_frac[2:] - jj_part[:2]) < tolerance_intersect or
                        np.linalg.norm(ii_frac[2:] - jj_part[2:]) < tolerance_intersect):

                    # Store the intersection coordinates in part_int
                    part_int[int_counter, :] = np.array([int_coord[0], int_coord[1]])
                    int_counter += 1

        # Add x0 and y0 of fracture ii to start of part_int, and x1 and y1 to the end of it.
        part_int = np.vstack((np.vstack((ii_frac[:2], part_int)), ii_frac[2:]))

        # Sort on x values
        part_int = part_int[np.lexsort((part_int[:, 1], part_int[:, 0]))]

        # Initialization of the array that will contain the information about the new fractures.
        new_fracs = np.zeros((num_ints+1, 5))
        for mm in range(0, num_ints + 1):
            x0, y0, x1, y1 = part_int[mm, 0], part_int[mm, 1], part_int[mm + 1, 0], part_int[mm + 1, 1]

            # This is how we find out in which subdomain the fracture will be. We add this ID to new_fracs
            subdomain_id = math.floor((((x0 + x1) / 2) - xmin) / interval_x) + \
                           math.floor((((y0 + y1) / 2) - ymin) / interval_y) * number_partitions_x

            new_fracs[mm, :] = np.array([x0, y0, x1, y1, subdomain_id])

        new_index += num_ints+1

        # Add fractures to the array that combines them all
        part_fracs[old_index:new_index] = new_fracs

        part_frac_order_vec[old_index:new_index] = np.ones(new_index - old_index) * frac_order_vec[fracs_to_part[ii]]

        old_index = new_index

    act_frac_sys = np.delete(act_frac_sys, fracs_to_part, axis=0)

    num_old_fracs = len(subdomain_sys[:, 0])

    subdomains_old = subdomain_sys[:, 0] + subdomain_sys[:, 1] * number_partitions_x

    subdomains_old = subdomains_old.reshape((num_old_fracs,1))

    act_frac_sys = np.hstack((act_frac_sys, subdomains_old))
    act_frac_sys = np.vstack((act_frac_sys, part_fracs))

    frac_order_vec = np.delete(frac_order_vec, fracs_to_part, axis=0)

    frac_order_vec = np.hstack((frac_order_vec, part_frac_order_vec))

    act_frac_sys_list = []
    frac_order_vec_list = []
    num_subdomains = number_partitions_x * number_partitions_y
    for p in range(0, num_subdomains):
        indices = np.where(act_frac_sys[:, 4] == p)
        act_frac_sys_list.append(act_frac_sys[indices, :4][0])
        frac_order_vec_list.append(frac_order_vec[indices])

    return act_frac_sys_list, frac_order_vec_list, partitions
