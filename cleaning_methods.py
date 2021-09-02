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
import numpy as np
import sys
from itertools import combinations
# from numba import jit
import os
from scipy import sparse
from scipy.sparse import csr_matrix


# @jit(nopython=True)
def segment_large_fracture(act_frac_sys, char_len, length_segms, order_discr, decimals):
    # Perform partitioning into smaller subsegments (around lc size):
    num_new_segms = int(np.sum(np.round(length_segms / char_len)) + np.sum(np.round(length_segms / char_len) == 0))
    act_frac_sys_new = np.zeros((num_new_segms, 4))
    order_discr_new = np.zeros((num_new_segms,))
    ith_segm = 0

    for ii in order_discr:
        size_segm = int(max(1, np.round(length_segms[ii] / char_len)))
        id_vec = np.arange(0, size_segm)
        act_frac_sys_new[ith_segm:(ith_segm + size_segm), 0] = act_frac_sys[ii, 0] + id_vec / size_segm * (
                    act_frac_sys[ii, 2] - act_frac_sys[ii, 0])
        act_frac_sys_new[ith_segm:(ith_segm + size_segm), 1] = act_frac_sys[ii, 1] + id_vec / size_segm * (
                    act_frac_sys[ii, 3] - act_frac_sys[ii, 1])
        act_frac_sys_new[ith_segm:(ith_segm + size_segm), 2] = act_frac_sys[ii, 0] + (id_vec + 1) / size_segm * (
                    act_frac_sys[ii, 2] - act_frac_sys[ii, 0])
        act_frac_sys_new[ith_segm:(ith_segm + size_segm), 3] = act_frac_sys[ii, 1] + (id_vec + 1) / size_segm * (
                    act_frac_sys[ii, 3] - act_frac_sys[ii, 1])

        # Update order of fractures:
        order_discr_new[ith_segm:(ith_segm + size_segm)] = np.arange(0, size_segm) + ith_segm

        ith_segm += size_segm

    act_frac_sys_new = np.round(act_frac_sys_new * 10 ** decimals) * 10 ** (-decimals)
    return act_frac_sys_new, order_discr_new


# @jit(nopython=True)
def create_graph(act_frac_sys_new, unique_nodes, order_discr_new, tolerance_zero, full_graph=False):
    # Store for each node to which segments it belongs and what its degree is:
    num_unq_nodes = unique_nodes.shape[0]
    num_segm_small = act_frac_sys_new.shape[0]
    incidence_matrix = np.zeros((num_segm_small * 2, 2), dtype=int)
    order_discr_nodes = np.ones((num_unq_nodes,), dtype=int) * (num_unq_nodes + 1)
    non_zero_count = 0

    for ii in range(num_unq_nodes):
        # Determine which segments the node belongs to (check unique both verses left en right node of each segment):
        ids_1, = np.where(((act_frac_sys_new[:, 0] - unique_nodes[ii, 0]) ** 2 +
                          (act_frac_sys_new[:, 1] - unique_nodes[ii, 1]) ** 2) ** (1/2) < tolerance_zero)
        ids_2, = np.where(((act_frac_sys_new[:, 2] - unique_nodes[ii, 0]) ** 2 +
                          (act_frac_sys_new[:, 3] - unique_nodes[ii, 1]) ** 2) ** (1/2) < tolerance_zero)
        connected_segms = np.union1d(ids_1, ids_2)

        # Store each nonzero entry:
        for jj in connected_segms:
            incidence_matrix[non_zero_count, :] = [ii, jj]
            non_zero_count += 1

        # Determine order of node:
        order_discr_nodes[ii] = int(np.min(np.append(order_discr_new[ids_1], order_discr_new[ids_2])))

    order_discr_nodes_new = np.argsort(order_discr_nodes)

    if full_graph:
        # Translate incidence matrix to sparse matrix:
        unique, node_degrees = np.unique(incidence_matrix[:, 0], return_counts=True)
        degree_matrix = sparse.diags(node_degrees)
        incidence_matrix_csr = csr_matrix((np.ones((incidence_matrix.shape[0])), (incidence_matrix[:, 0], incidence_matrix[:, 1])))
        adjacency_matrix_csr = np.dot(incidence_matrix_csr, incidence_matrix_csr.T) - degree_matrix
        laplacian_matrix = degree_matrix - adjacency_matrix_csr
        return incidence_matrix, order_discr_nodes_new, degree_matrix, adjacency_matrix_csr, laplacian_matrix
    else:
        return incidence_matrix, order_discr_nodes_new

def reorder_incidence_and_nodes(incidence_matrix, unique_nodes, order_discr):
    incidence_matrix = incidence_matrix[np.where(incidence_matrix[:, 0] != -99999)[0], :]

    unique_node_list = np.unique(incidence_matrix[:, 0])
    unique_segm_list = np.unique(incidence_matrix[:, 1])

    node_mapping = np.ones((unique_node_list.shape[0],), dtype=int) * (-1)
    segm_mapping = np.ones((unique_segm_list.shape[0],), dtype=int) * (-1)

    incidence_matrix_new = np.zeros((incidence_matrix.shape[0], 2), dtype=int)
    node_counter = 0
    segm_counter = 0

    for ii, row in enumerate(incidence_matrix):
        # if not np.equal(row, incidence_matrix_new).all(axis=1).any():
        #     # The
        if row[0] not in node_mapping:
            node_mapping[node_counter] = row[0]
            node_counter += 1

        if row[1] not in segm_mapping:
            segm_mapping[segm_counter] = row[1]
            segm_counter += 1

        # Add entry to the incidence list:
        incidence_matrix_new[ii, :] = [np.where(node_mapping == row[0])[0][0], np.where(segm_mapping == row[1])[0][0]]

    unique_nodes_new = unique_nodes[node_mapping, :]

    # Re-map the order of the node back to the new ordering:
    order_discr_new = np.zeros((unique_nodes_new.shape[0],), dtype=int)

    for ii, node in enumerate(order_discr):
        order_discr_new[ii] = np.where(node == node_mapping)[0]

    return incidence_matrix_new, unique_nodes_new, order_discr_new


# @jit(nopython=True)
def closest_point_method(incidence_matrix, char_len, order_discr_nodes, unique_nodes, merge_threshold):
    # Loop over all unique nodes (except the first):
    num_unq_nodes = unique_nodes.shape[0]
    order_mask = np.ones((num_unq_nodes,), dtype=bool)
    count = 0
    for new_node in order_discr_nodes[1:]:
        # Check if node is within lc/2 radius of any [1, ..., N] node:
        # Then select smallest distance if true and merge incidence matrix and distance for that node

        count += 1
        # Compute distance matrix:
        # id_min_node = np.argmin(dist_mat_order[ii, :ii])
        dist_vec_temp = np.linalg.norm(unique_nodes[new_node, :] - unique_nodes[order_discr_nodes[:count], :], axis=1)
        argmin_id = np.argmin(dist_vec_temp)
        fixed_node = order_discr_nodes[argmin_id]

        if dist_vec_temp[argmin_id] < (char_len * merge_threshold):
            # Record shift in order of node-importance:
            order_mask[np.where(order_discr_nodes == new_node)[0]] = False

            # Todo: Check if merging nodes is still compatible with old way of solving
            # Todo: What does it mean to take the union of the rows of the incidence matrix in the case of the list approach
            # Todo: Simply replacing the node with new node into which it merges still doesn't take care of the segments that might get merged
            # Todo: How to identify merged and collapsed segments in this new approach?
            # Take the union of incidency matrix (this preserves connection that existed at the node which is removed,
            # basically it means, keep all remaining connections and add new ones due to new node merge):
            # In the list approach, this means updating the entries which the new_node was part of with the vertex of
            # the fixed node:
            old_connections = incidence_matrix[:, 0] == new_node
            incidence_matrix[old_connections, 0] = fixed_node
            edges_to_node = incidence_matrix[old_connections, 1]

            # Now check for all those entries if the node has been collapsed, merged or simply extended:
            for edge in edges_to_node:
                nodes_on_edge = incidence_matrix[np.where(incidence_matrix[:, 1] == edge)[0], 0]

                if nodes_on_edge.shape[0] != 2:
                    print('Edge has more than two vertices, something is wrong here!!!')
                    sys.exit(0)

                if nodes_on_edge[0] == nodes_on_edge[1]:
                    # Means that the segment has collapsed and therefore has zero length --> needs to be removed!
                    incidence_matrix[np.where(incidence_matrix[:, 1] == edge)[0], :] = -99999

                else:
                    # Check if the intersection of the set of edges on the two node exceeds 1:
                    # Two nodes can only share an edge if they are ON(!) the same edge, this means that if there are two
                    # edges that satisfy this condition, there is a double/merged segment and need to keep one segment
                    # and remove the other(s)!
                    edges_1 = incidence_matrix[np.where(incidence_matrix[:, 0] == nodes_on_edge[0])[0], 1]
                    edges_2 = incidence_matrix[np.where(incidence_matrix[:, 0] == nodes_on_edge[1])[0], 1]
                    common_edges = np.intersect1d(edges_1, edges_2)

                    if common_edges.shape[0] > 1:
                        # Remove all edges in excess of 1:
                        for remove_edge in common_edges[1:]:
                            incidence_matrix[np.where(incidence_matrix[:, 1] == remove_edge)[0], :] = -99999

            unique_nodes[new_node, :] = -99999  # node doesn't exist anymore!

    # Make sure ordering of nodes is preserved
    order_discr_nodes = order_discr_nodes[order_mask]
    incidence_matrix, unique_nodes, order_discr_nodes = reorder_incidence_and_nodes(incidence_matrix, unique_nodes, order_discr_nodes)

    return incidence_matrix, unique_nodes, order_discr_nodes


# @jit(nopython=True)
def straighten_fractures(incidence_matrix, unique_nodes, angle_tol_straighten, order_discr_nodes):
    unique_node_tot = unique_nodes.shape[0]
    unique_node_list, degree_nodes = np.unique(incidence_matrix[:, 0], return_counts=True)
    unique_edges, counts = np.unique(incidence_matrix[:, 1], return_counts=True)
    num_segm_tot = unique_edges.shape[0]

    # Straighten fractures:
    # Take only the degree-2 nodes to straighten:
    nodes_to_straighten = unique_node_list[np.where(degree_nodes == 2)[0]]
    order_mask = np.ones((unique_node_tot,), dtype=bool)
    for ii, node in enumerate(nodes_to_straighten):
        # Check the angle between two-segments that this node is on:
        segms = incidence_matrix[np.where(incidence_matrix[:, 0] == node)[0], 1]
        if len(segms) > 2:
            a = 0
        nodes_segm1 = incidence_matrix[np.where(incidence_matrix[:, 1] == segms[0])[0], 0]
        nodes_segm2 = incidence_matrix[np.where(incidence_matrix[:, 1] == segms[1])[0], 0]
        vec_m = unique_nodes[nodes_segm1[0], :] - unique_nodes[nodes_segm1[1], :]
        vec_p = unique_nodes[nodes_segm2[0], :] - unique_nodes[nodes_segm2[1], :]
        dot_product = min(1, max(-1, np.dot(vec_m / np.linalg.norm(vec_m), vec_p / np.linalg.norm(vec_p))))
        angle = np.arccos(np.abs(dot_product)) * 180 / np.pi

        # If the angle is below threshold, add node to remove nodes and make sure to take the union of
        if angle < angle_tol_straighten:
            # Always keep segms[0] and remove segms[1] (removing segment in this context is equal to saying the segment
            # doesn't contain any nodes --> setting col segms[1] to zeros, also remember to set row to zeros of node which
            # we remove (and finally record which cols and rows to delete)
            order_mask[np.where(order_discr_nodes == node)[0]] = False

            # Remove node to straight edges --> means from 2 segments to 1 --> 1 segment goes from node [i, j] and
            # [j, k] to [i, k] basically: take union, remove common node --> make one segm these nodes, remove the other
            # segment with -99999
            incidence_matrix[np.where(incidence_matrix[:, 0] == node)[0], :] = -99999

            # Keep segms[0] comes down to adding segms[0] to other node segms[1], where other means not the shared node!
            # It means finding row of incidence matrix with contains the node nodes_segm2 != node and the segm segms[1]
            row_id = np.intersect1d(np.where(incidence_matrix[:, 0] == nodes_segm2[nodes_segm2 != node])[0],
                                    np.where(incidence_matrix[:, 1] == segms[1])[0])
            incidence_matrix[row_id, 1] = segms[0]

    order_discr_nodes = order_discr_nodes[order_mask]
    incidence_matrix, unique_nodes, order_discr_nodes = reorder_incidence_and_nodes(incidence_matrix, unique_nodes, order_discr_nodes)

    return incidence_matrix, unique_nodes, order_discr_nodes


# @jit(nopython=True)
def remove_small_angles(incidence_matrix, unique_nodes, tolerance_zero, angle_tol_remove_segm, order_discr_nodes):
    unique_edges, counts = np.unique(incidence_matrix[:, 1], return_counts=True)
    num_segm_tot = unique_edges.shape[0]
    new_unique_node_tot = unique_nodes.shape[0]
    merge_list = np.ones((new_unique_node_tot,), dtype=int) * -1
    order_mask = np.ones((new_unique_node_tot,), dtype=bool)
    merge_list_segm = np.ones((num_segm_tot,), dtype=int) * -1

    for ii in range(new_unique_node_tot):
        if merge_list[ii] < 0:  # note sure if this is necessary with the new checks!!! todo: check tomorrow if necessary
            # Determine which segments belong to the current node:
            segms = incidence_matrix[np.where(incidence_matrix[:, 0] == ii)[0], 1]

            # Remove segments from segms-list that have been removed/merged:
            segms = segms[np.where(merge_list_segm[segms] < 0)[0]]

            # Loop over all segment combinations to determine the angle between them:
            permutations = list(combinations(segms, 2))
            angles_perm = np.ones((len(permutations),)) * 180
            for jj, segm_pair in enumerate(permutations):
                # Removed segments are either segments that after merging have only one node left (since the original node is
                # merged and therefore doesn't exist anymore in the incidence matrix) (this check is necessary because a
                # segment that is merged like this can happen outside of the two segments that are investiated on their angle,
                # it can be attached to the node which is getting merged and another node)
                removed_segm = False
                nodes_semg1 = incidence_matrix[np.where(incidence_matrix[:, 1] == segm_pair[0])[0], 0]
                nodes_semg2 = incidence_matrix[np.where(incidence_matrix[:, 1] == segm_pair[1])[0], 0]

                if len(nodes_semg1) < 2:
                    # Segment has been removed:
                    merge_list_segm[segm_pair[0]] = 1
                    incidence_matrix[np.where(incidence_matrix[:, 1] == segm_pair[0])[0], :] = -99999
                    removed_segm = True

                if len(nodes_semg2) < 2:
                    # Segment has been removed:
                    merge_list_segm[segm_pair[1]] = 1
                    incidence_matrix[np.where(incidence_matrix[:, 1] == segm_pair[1])[0], :] = -99999
                    removed_segm = True

                if not removed_segm:
                    # If the segment is not removed (removed meaning merged into another segment and it's original node is
                    # removed which causes it to have less than two nodes):
                    vec_m = unique_nodes[ii, :] - unique_nodes[nodes_semg1[nodes_semg1 != ii][0], :]
                    vec_p = unique_nodes[ii, :] - unique_nodes[nodes_semg2[nodes_semg2 != ii][0], :]
                    dot_product = min(1, max(-1, np.dot(vec_m / np.linalg.norm(vec_m), vec_p / np.linalg.norm(vec_p))))
                    angles_perm[jj] = np.arccos(dot_product) * 180 / np.pi
                else:
                    angles_perm[jj] = 180

            for jj in range(angles_perm.shape[0]):
                if angles_perm[jj] < np.max([tolerance_zero, 1e-5]):
                    # If angle is zero, means we have a segment which is overlapping, set angle to 180 and set column of
                    # incidence matrix to zero (for one of the segments):
                    segm_to_remove = permutations[jj][0]  # since segments are overlapping, no difference between segments
                    incidence_matrix[np.where(incidence_matrix[:, 1] == segm_to_remove)[0], :] = -99999
                    merge_list_segm[segm_to_remove] = 1
                    angles_perm[jj] = 180

                    # Also set angles for all other pairs with this segment to 180:
                    for kk in permutations:
                        if segm_to_remove == kk[0] or segm_to_remove == kk[1]:
                            angles_perm[jj] = 180

            if any(angles_perm < angle_tol_remove_segm):
                # There exists a very small angle between some of the segments in the array:
                # Determine the smallest angle and solve the issue, fix the issue and recalculate the angles
                id_small_pair = np.argmin(angles_perm)

                nodes_semg1 = incidence_matrix[np.where(incidence_matrix[:, 1] == permutations[id_small_pair][0])[0], 0]
                nodes_semg2 = incidence_matrix[np.where(incidence_matrix[:, 1] == permutations[id_small_pair][1])[0], 0]

                vec_m = unique_nodes[ii, :] - unique_nodes[nodes_semg1[nodes_semg1 != ii][0], :]
                vec_p = unique_nodes[ii, :] - unique_nodes[nodes_semg2[nodes_semg2 != ii][0], :]
                length_m = np.linalg.norm(vec_m)
                length_p = np.linalg.norm(vec_p)

                # Merge non-shared node of smaller segment in one of the nodes of the larger segment (the closest node to be
                # precise)
                if length_m < length_p:
                    merge_node_id = nodes_semg1[nodes_semg1 != ii][0]
                else:
                    merge_node_id = nodes_semg2[nodes_semg2 != ii][0]

                possible_merge_ids = np.union1d(nodes_semg1, nodes_semg2)
                possible_merge_ids = possible_merge_ids[possible_merge_ids != merge_node_id]

                # Determine which node to merge into (the closest node of the large segment):
                dist_nodes = np.linalg.norm(unique_nodes[possible_merge_ids, :] - unique_nodes[merge_node_id, :],
                                            axis=1)
                merge_target_id = possible_merge_ids[np.argmin(dist_nodes)]

                # Record also the node which it gets merged into, this is important to preserve connections in incidence matrix)
                merge_list[merge_node_id] = merge_target_id
                order_mask[np.where(order_discr_nodes == merge_node_id)[0]] = False

                old_connections = incidence_matrix[:, 0] == merge_node_id
                incidence_matrix[old_connections, 0] = merge_target_id
                edges_to_node = incidence_matrix[old_connections, 1]

                # Now check for all those entries if the node has been collapsed, merged or simply extended:
                for edge in edges_to_node:
                    nodes_on_edge = incidence_matrix[np.where(incidence_matrix[:, 1] == edge)[0], 0]

                    if nodes_on_edge.shape[0] != 2:
                        print('Edge has more than two vertices, something is wrong here!!!')
                        sys.exit(0)

                    if nodes_on_edge[0] == nodes_on_edge[1]:
                        # Means that the segment has collapsed and therefore has zero length --> needs to be removed!
                        incidence_matrix[np.where(incidence_matrix[:, 1] == edge)[0], :] = -99999

                    else:
                        # Check if the intersection of the set of edges on the two node exceeds 1:
                        # Two nodes can only share an edge if they are ON(!) the same edge, this means that if there are two
                        # edges that satisfy this condition, there is a double/merged segment and need to keep one segment
                        # and remove the other(s)!
                        edges_1 = incidence_matrix[np.where(incidence_matrix[:, 0] == nodes_on_edge[0])[0], 1]
                        edges_2 = incidence_matrix[np.where(incidence_matrix[:, 0] == nodes_on_edge[1])[0], 1]
                        common_edges = np.intersect1d(edges_1, edges_2)

                        if common_edges.shape[0] > 1:
                            # Remove all edges in excess of 1:
                            for remove_edge in common_edges[1:]:
                                incidence_matrix[np.where(incidence_matrix[:, 1] == remove_edge)[0], :] = -99999
                                merge_list_segm[remove_edge] = 1

                unique_nodes[merge_node_id, :] = -99999  # node doesn't exist anymore!

    # Make sure ordering of nodes is preserved
    order_discr_nodes = order_discr_nodes[order_mask]
    incidence_matrix, unique_nodes, order_discr_nodes = reorder_incidence_and_nodes(incidence_matrix, unique_nodes,
                                                                                    order_discr_nodes)
    return incidence_matrix, unique_nodes, order_discr_nodes

# @jit(nopython=True)
def reconstruct_act_frac_sys(incidence_matrix, unique_nodes):
    # Reconstruct the m x 4 matrix containing each segment of the fracture network:
    unique_segm_list, count_segment = np.unique(incidence_matrix[:, 1], return_counts=True)
    segm_count = 0
    act_frac_sys = np.zeros((unique_segm_list.shape[0], 4))
    for ii in unique_segm_list:
        nodes_on_segm = incidence_matrix[np.where(incidence_matrix[:, 1] == ii)[0], 0]

        if nodes_on_segm.shape[0] != 2:
            print('THERE IS A SEGMENT WITH MORE THAN 2 NODES!!!')
            sys.exit(0)

        act_frac_sys[segm_count, :] = np.hstack((unique_nodes[nodes_on_segm[0], :],
                                                 unique_nodes[nodes_on_segm[1], :]))
        segm_count += 1
    return act_frac_sys


def write_new_frac_sys(act_frac_sys_new, filename):
    f = open(filename, "w+")
    for frac in act_frac_sys_new:
        f.write('{:9.5f} {:9.5f} {:9.5f} {:9.5f}\n'.format(frac[0], frac[1], frac[2], frac[3]))
    f.close()
    return 0


def create_geo_file(act_frac_sys, unique_nodes, incidence_matrix, filename, decimals,
                    height_res, char_len, box_data, char_len_boundary):
    act_frac_sys = np.round(act_frac_sys * 10 ** decimals) * 10 ** (-decimals)
    num_segm_tot = act_frac_sys.shape[0]
    num_nodes_tot = unique_nodes.shape[0]
    f = open(filename, "w+")

    # Note: always end statement in GMSH with ";"
    # Note: comments in GMSH are as in C(++) "//"
    f.write('// Geo file which meshes the input mesh from act_frac_sys.\n')
    f.write('// Change mesh-elements size by varying "lc" below.\n\n')

    # Can specify the type of meshing algorithm for 2D meshing here:
    # f.write('// Specify meshing algorithm:\n')
    # f.write('-algo meshadapt;\n\n')

    # Set some parameters in the model:
    f.write('lc = {:1.3f};\n'.format(char_len))
    f.write('lc_box = {:1.3f};\n'.format(char_len_boundary))
    f.write('height_res = {:4.3f};\n\n'.format(height_res))

    # Allocate memory for points_created array and counters:
    points_created = np.zeros((num_nodes_tot,))
    line_count = 0
    point_count = 0

    unique_segm_list, segm_count = np.unique(incidence_matrix[:, 1], return_counts=True)

    for ii in unique_segm_list:
        # Take two points per segment and write to .geo file:
        # e.g. point: Point(1) = {.1, 0, 0, lc};
        nodes = incidence_matrix[np.where(incidence_matrix[:, 1] == ii)[0], 0]

        if len(nodes) != 2:
            print('There exists a segment with an amount of nodes unequal to 2!!!')
            sys.exit(0)

        # Check if first point is already created, if not, add it:
        if points_created[nodes[0]] == 0:
            points_created[nodes[0]] = 1
            point_count += 1
            f.write('Point({:d}) = {{{:8.5f}, {:8.5f}, 0, lc}};\n'.format(nodes[0] + 1,
                                                                           unique_nodes[nodes[0], 0],
                                                                           unique_nodes[nodes[0], 1]))

        if points_created[nodes[1]] == 0:
            points_created[nodes[1]] = 1
            point_count += 1
            f.write('Point({:d}) = {{{:8.5f}, {:8.5f}, 0, lc}};\n'.format(nodes[1] + 1,
                                                                           unique_nodes[nodes[1], 0],
                                                                           unique_nodes[nodes[1], 1]))

        line_count += 1
        f.write('Line({:d}) = {{{:d}, {:d}}};\n\n'.format(line_count, nodes[0] + 1, nodes[1] + 1))

    # Store some internal variables for gmsh (used later after extrude):
    f.write('num_points_frac = newp - 1;\n')
    f.write('num_lines_frac = newl - 1;\n\n')

    # Write the box_data (box around fracture network in which we embed the fractures)
    f.write('// Extra points for boundary of domain:\n')
    for ii in range(4):
        # For every corner of the box:
        point_count += 1
        f.write('Point({:d}) = {{{:8.5f}, {:8.5f}, 0, lc_box}};\n'.format(point_count,
                                                                          box_data[ii, 0], box_data[ii, 1]))

    # Add four lines for each side of the box:
    f.write('\n// Extra lines for boundary of domain:\n')
    line_count += 1
    f.write('Line({:d}) = {{{:d}, {:d}}};\n'.format(line_count, point_count - 3, point_count - 2))

    line_count += 1
    f.write('Line({:d}) = {{{:d}, {:d}}};\n'.format(line_count, point_count - 2, point_count - 1))

    line_count += 1
    f.write('Line({:d}) = {{{:d}, {:d}}};\n'.format(line_count, point_count - 1, point_count - 0))

    line_count += 1
    f.write('Line({:d}) = {{{:d}, {:d}}};\n'.format(line_count, point_count - 0, point_count - 3))

    # Make Curve loop for the boundary:
    f.write('\n// Create line loop for boundary surface:\n')
    f.write('Curve Loop(1) = {{{:d}, {:d}, {:d}, {:d}}};\n'.format(line_count - 3, line_count - 2,
                                                                   line_count - 1, line_count))
    f.write('Plane Surface(1) = {1};\n\n')
    f.write('Curve{1:num_lines_frac} In Surface{1};\n')

    # Extrude model to pseuo-3D:
    f.write('\n// Extrude surface with embedded features\n')
    f.write('Extrude {0, 0, height_res}{ Surface {1}; Layers{1}; Recombine;}\n')
    f.write('Physical Volume("matrix", 9991) = {1};\n')

    f.write('num_surfaces_before = news;\n')
    f.write('Extrude {0, 0, height_res}{ Line {1:num_lines_frac}; Layers{1}; Recombine;}\n')
    f.write('num_surfaces_after = news - 1;\n')
    f.write('num_surfaces_fracs = num_surfaces_after - num_surfaces_before;\n\n')
    f.write('Physical Surface("fracture", 99991) = {num_surfaces_before:num_surfaces_after};\n')

    # Create mesh and perform coherency check:
    f.write('Mesh 3;  // Generalte 3D mesh\n')
    f.write('Coherence Mesh;  // Remove duplicate entities\n')
    f.write('Mesh.MshFileVersion = 2.1;\n')
    f.close()
    return 0


def mesh_geo_file(filename_in, filename_out):
    # subprocess.call(['C:\\Temp\\a b c\\Notepad.exe', 'C:\\test.txt'])
    # subprocess.call("'gmsh ' {:s} ' -o ' {:s} ' -save'".format(filename_in, filename_out))
    os.system("gmsh {:s} -o {:s} -save".format(filename_in, filename_out))
    return 0


def extract_unique_nodes(act_frac_sys, remove_small_segms=True, tolerance_zero=1e-5):
    # Extract unique indices:
    dummy_sys = np.array(act_frac_sys, copy=True)
    indices = np.where(act_frac_sys[:, 0] > act_frac_sys[:, 2])[0]
    dummy_sys[indices, 0:2] = act_frac_sys[indices, 2:]
    dummy_sys[indices, 2:] = act_frac_sys[indices, :2]
    dummy_sys = np.unique(dummy_sys, axis=0)

    if remove_small_segms:
        len_segm = np.sqrt((dummy_sys[:, 0] - dummy_sys[:, 2])**2 + (dummy_sys[:, 1] - dummy_sys[:, 3])**2)
        dummy_sys = dummy_sys[len_segm > tolerance_zero, :]
    return dummy_sys


# Full cleaning method:
# @jit(nopython=True)
def full_cleaning(act_frac_sys, char_len, decimals, tolerance_zero, angle_tol_straighten,
                  angle_tol_remove_segm, straighten_first, merge_threshold):
    act_frac_sys = np.round(act_frac_sys * 10 ** decimals) * 10 ** (-decimals)

    # Calculate the length of segments:
    length_segms = ((act_frac_sys[:, 0] - act_frac_sys[:, 2]) ** 2 +
                    (act_frac_sys[:, 1] - act_frac_sys[:, 3]) ** 2) ** (1 / 2)
    order_discr = np.argsort(-length_segms)  # larger fractures first, smaller fractures last

    # Perform segmentation large fractures into fracture segments of length char_len:
    act_frac_sys_segm, order_discr_segm = segment_large_fracture(act_frac_sys, char_len, length_segms, order_discr,
                                                                 decimals)

    # Find unique nodes:
    unique_nodes = np.unique(np.vstack((act_frac_sys_segm[:, :2], act_frac_sys_segm[:, 2:])), axis=0)

    incidence_matrix, order_discr_nodes = create_graph(act_frac_sys_segm, unique_nodes, order_discr_segm,
                                                           tolerance_zero)

    # # First calculate distance matrix
    # dist_vec_order = pdist(unique_nodes[order_discr_nodes, :], metric='euclidean')
    # dist_mat_order = squareform(dist_vec_order)

    num_inner_iter = 2
    for kk in range(num_inner_iter):
        # Perform merging algorithm (closest point method):
        incidence_matrix, unique_nodes, order_discr_nodes = \
            closest_point_method(incidence_matrix, char_len, order_discr_nodes, unique_nodes, merge_threshold)

    # The order of these two matters (not sure how much, todo: check how much the order influences the results):
    remove_angle_iters = 5
    if straighten_first:
        # Perform straightening of fractures:
        incidence_matrix, unique_nodes, order_discr_nodes = \
            straighten_fractures(incidence_matrix, unique_nodes, angle_tol_straighten, order_discr_nodes)

        # Remove small angle fracture intersections:
        for kk in range(remove_angle_iters):
            incidence_matrix, unique_nodes, order_discr_nodes = \
                remove_small_angles(incidence_matrix, unique_nodes, tolerance_zero, angle_tol_remove_segm,
                                    order_discr_nodes)
    else:
        # Remove small angle fracture intersections:
        for kk in range(remove_angle_iters):
            incidence_matrix, unique_nodes, order_discr_nodes = \
                remove_small_angles(incidence_matrix, unique_nodes, tolerance_zero, angle_tol_remove_segm,
                                    order_discr_nodes)

        # Perform straightening of fractures:
        incidence_matrix, unique_nodes, order_discr_nodes = \
            straighten_fractures(incidence_matrix, unique_nodes, angle_tol_straighten, order_discr_nodes)

    # Re-calculate act_frac_sys from incidence matrix and unique node array:
    act_frac_sys = reconstruct_act_frac_sys(incidence_matrix, unique_nodes)

    return act_frac_sys, unique_nodes, incidence_matrix