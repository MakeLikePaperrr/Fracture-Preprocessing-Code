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
import matplotlib.pyplot as plt
import igraph
import copy
import matplotlib.colors as colors
import matplotlib.cm as cm
import os


class Graph:
    """
        (weighted) Graph class containing a list of Vertices (x- and y-coordinates) and Edges (list of
        (v_i, v_j) pairs), where v_i and v_j are vertex i and j respectively)
    """
    def __init__(self, max_vertices=int(1e5), max_edges=int(1e5), edge_data=None, matrix_perm=1000, fracture_aperture=1e-4):
        self.vertices = np.ones((max_vertices, 2), dtype=float) * np.inf
        self.edge_to_vertex = np.ones((max_edges, 2), dtype=int) * -1
        self.apertures = np.ones((max_edges,), dtype=float) * fracture_aperture
        self.heat_transfer_mult = np.ones((max_edges,), dtype=float)
        self.volumes = np.zeros((max_edges,), dtype=float)
        self.active_edges = np.zeros((max_edges,), dtype=bool)
        self.active_vertices = np.zeros((max_vertices,), dtype=bool)
        self.vertex_to_edge = []
        self.num_edges = 0
        self.matrix_perm = matrix_perm * 1e-15
        self.matrix_eff_aperture = np.sqrt(12 * self.matrix_perm)

        if edge_data is not None:
            # Construct graph based on edge_data:
            self.add_multiple_edges(edge_data)

    def add_edge(self, x1, y1, x2, y2):
        """
        Add edge to self.edge_to_vertex matrix [vertex_id_i, vertex_id_j], and adds new edge to self.vertex_to_edge for
        vertex_id_i and vertex_id_j
        :param x1: x-coordinate of vertex_id_i
        :param y1: y-coordinate of vertex_id_i
        :param x2: x-coordinate of vertex_id_j
        :param y2: y-coordinate of vertex_id_j
        :return:
        """
        # Find or insert both vertices from edge:
        vertex_id_1 = self.find_or_insert_vertex(x1, y1)
        vertex_id_2 = self.find_or_insert_vertex(x2, y2)

        if vertex_id_1 == vertex_id_2:
            return 0

        edge_id = self.num_edges
        self.edge_to_vertex[edge_id, 0] = vertex_id_1
        self.edge_to_vertex[edge_id, 1] = vertex_id_2
        self.num_edges += 1
        self.active_edges[edge_id] = True
        self.volumes[edge_id] = self.apertures[edge_id] * np.linalg.norm(self.vertices[vertex_id_1].flatten() -
                                                                         self.vertices[vertex_id_2].flatten())

        self.vertex_to_edge[vertex_id_1].append(edge_id)
        self.vertex_to_edge[vertex_id_2].append(edge_id)
        return 0

    def add_multiple_edges(self, edge_data):
        """
        Loop over edgedata to construct graph
        :param edge_data: N x 4 array that contains all the edges to be added to the domain, for should be:
        [[x1, y1, x2, y2], ..., []]
        :return:
        """
        for edge in edge_data:
            self.add_edge(edge[0], edge[1], edge[2], edge[3])
        return 0

    def get_num_vertices(self):
        return len(self.vertex_to_edge)

    def get_num_edges(self):
        return self.num_edges

    def insert_vertex(self, x: float, y: float):
        """
        Insert new vertex into self.vertices and update number of existing vertices
        NOTE: This method currently assumes that the vertex with coordinates (x,y) is not yet in self.vertices!!!
        :param x: x-coordinate of vertex
        :param y: y-coordinate of vertex
        :return: the new id of vertex
        """
        vertex_id = self.get_num_vertices()
        self.vertices[vertex_id, 0] = x
        self.vertices[vertex_id, 1] = y
        self.vertex_to_edge.append([])
        self.active_vertices[vertex_id] = True
        return vertex_id

    def get_vertex_id(self, x, y):
        """
        Get vertex id if already in the domain, otherwise return 0, also catches if there exists duplicated vertices in
        the domain (i.e., when results.size > 1)
        :param x: x-coordinate of vertex
        :param y: y-coordinate of vertex
        :return:
        """
        # linear search is a potential problem
        # need to introduce additional, sorted by coordinate array, coord_to_vertex_id
        result = np.where((self.vertices[0:self.get_num_vertices()] == (x, y)).all(axis=1))[0]
        # assert (result.size == 1)
        if result.size == 1:
            return result[0]
        elif result.size == 0:
            return None
        else:
            # Duplicate vertex...
            print('Duplicate vertex found in self.get_vertex_id with coordinates [x,y] = [{:}, {:}]'.format(x,y))
            return np.NaN

    def find_or_insert_vertex(self, x, y):
        """
        Flexible method which finds the id of an existing vertex or adds a potentially new vertex to the graph
        :param x: x-coordinate of vertex
        :param y: y-coordinate of vertex
        :return:
        """
        vertex_id = self.get_vertex_id(x, y)
        if vertex_id is None:
            # New vertex:
            vertex_id = self.insert_vertex(x, y)
        return vertex_id

    def remove_edges(self, ghost_edge_to_vertices, ghosted_edges):
        """
        Removes edges that were flagged as ghosts (edge_to_vertex remains unchanged in size, to simplify bookkeeping,
        edges are removed from vertex_to_edge to keep node degree accurate
        :param ghost_edge_to_vertices: list of vertices for each edge that is removed after merging vertex
        :param ghosted_edges: list of edges that are removed after merging vertex
        :return:
        """
        for ii in range(len(ghosted_edges)):
            edge = ghosted_edges[ii]
            for jj in ghost_edge_to_vertices[ii]:
                self.vertex_to_edge[jj].remove(edge)

            self.edge_to_vertex[edge] = -1
            self.active_edges[edge] = False

    def find_conflicting_edges(self, vertex_id_from, vertex_id_to):
        """
        For each edge leaving vertex_from, check what is the status after merging and flag for ghosting if edge is
        collapsed or overlaps after merging
        :param vertex_id_from: vertex that is removed after merging
        :param vertex_id_to: vertex that is merged into
        :return:
        """
        ghosted_edges = []
        ghost_edge_to_vertices = []
        status_after_merge = [None] * len(self.vertex_to_edge[vertex_id_from])
        edge_id_after_merge = [None] * len(self.vertex_to_edge[vertex_id_from])
        edge_to_vertex_after_merge = np.ones((len(self.vertex_to_edge[vertex_id_from]), 2), dtype=int) * -1
        edges = self.vertex_to_edge[vertex_id_from]
        for ii, edge in enumerate(edges):
            status_after_merge[ii], edge_to_vertex_after_merge[ii], edge_id_after_merge[ii] = \
                self.status_edge_after_merge(vertex_id_from, vertex_id_to, edge)
            if status_after_merge[ii] == 'collapsed' or status_after_merge[ii] == 'overlap':
                # Edge will be ghosted:
                ghosted_edges.append(edge)
                ghost_edge_to_vertices.append(list(self.edge_to_vertex[edge]))

        # self.visualize_sub_graph(self.edge_to_vertex[list(set(self.vertex_to_edge[vertex_id_from] + self.vertex_to_edge[vertex_id_to]))])
        return ghost_edge_to_vertices, ghosted_edges, status_after_merge, edge_to_vertex_after_merge, edge_id_after_merge

    def merge_vertices(self, vertex_id_from, vertex_id_to, char_len, correct_aperture):
        """
        Merge two vertices in domain while preserving connections and ghosting the merged vertex
        :param vertex_id_from: vertex that is merged in vertex_id_to (is ghosted)
        :param vertex_id_to: target vertex which vertex_id_from is merged into (stays in domain)
        :return:
        """
        # ghost any edge that "collapses" (i.e., the edge is shared by both vertices and has zero length after
        # vertex-merging) or if the edge will overlap after merging.
        ghost_edge_to_vertices, ghosted_edges, status_after_merging, edge_to_vertex_after_merging, edge_id_after_merge = \
            self.find_conflicting_edges(vertex_id_from, vertex_id_to)
        if correct_aperture:
            self.update_volumes_and_apertures(vertex_id_from, vertex_id_to, status_after_merging, edge_to_vertex_after_merging, edge_id_after_merge, char_len)
        self.remove_edges(ghost_edge_to_vertices, ghosted_edges)

        # 1. preserve connections to ghost-vertex
        [self.vertex_to_edge[vertex_id_to].append(x) for x in self.vertex_to_edge[vertex_id_from] if
            x not in self.vertex_to_edge[vertex_id_to]]
        # self.vertex_to_edge[vertex_id_to] = set(self.vertex_to_edge[vertex_id_to] + self.vertex_to_edge[vertex_id_from])

        # 2. apply ghosting to vertex:
        self.vertices[vertex_id_from] = np.inf
        self.active_vertices[vertex_id_from] = False

        # 3. update all instances of ghost-vertex in edge_to_vertex to new vertex_id
        slice_edge_to_vertex = self.edge_to_vertex[0:self.get_num_edges()]
        slice_edge_to_vertex[slice_edge_to_vertex == vertex_id_from] = vertex_id_to
        return 0

    def check_distance_constraint(self, new_vertex, existing_vertices, char_len, merge_threshold):
        """
        Computes the distance between all vertices already in domain and to be added vertex
        :param new_vertex: to be added vertex
        :param existing_vertices: vertices already approved
        :param char_len: radius for algebraic constraint
        :param merge_threshold: h-factor which is recommended between [0.5, 0.86]
        :return:
        """
        assert 0.5 <= merge_threshold <= 0.86, "Choose threshold on closed interval [0.5, 0.86]"
        dist_vec = np.linalg.norm(self.vertices[new_vertex] - self.vertices[existing_vertices], axis=1)
        argmin_id = np.argmin(dist_vec)
        fixed_vertex = existing_vertices[argmin_id]

        if dist_vec[argmin_id] < char_len * merge_threshold:
            return fixed_vertex
        else:
            return -1

    def closest_point_method(self, order_discr, char_len, merge_threshold, correct_aperture):
        """
        Main sequential algorithm which performs cleaning of the fracture network based on algebraic constraint
        :param order_discr: order in which vertices are sequentially added to the domain and checked for constraint
        :param char_len: radius within which vertices are violating constraint
        :param merge_threshold: h-factor which is recommended between [0.5, 0.86]
        :param correct_aperture: boolean for applying aperture correction or not
        :return:
        """
        assert 0.5 <= merge_threshold <= 0.86, "Choose threshold on closed interval [0.5, 0.86]"
        count = 0
        for new_vertex in order_discr[1:]:
            count += 1
            if not self.active_vertices[new_vertex]:
                continue
            fixed_vertex = self.check_distance_constraint(new_vertex, order_discr[:count], char_len, merge_threshold)
            if fixed_vertex >= 0:
                self.merge_vertices(new_vertex, fixed_vertex, char_len, correct_aperture)
        return 0

    def calc_angles_vectors_2D(self, vec_m, vec_p, angle_360=False):
        """
        Computes the angle between any two vectors in 2D
        :param vec_m: first vector
        :param vec_p: second vector
        :param angle_360: boolean for returning angle in 360 or 180 180 degrees spectrum
        :return:
        """
        vec_m = vec_m / np.linalg.norm(vec_m)
        vec_p = vec_p / np.linalg.norm(vec_p)
        dot_product = min(1, max(-1, np.dot(vec_m, vec_p)))

        if angle_360:
            det_product = vec_m[0] * vec_p[1] - vec_m[1] * vec_p[0]
            angle_full = np.arctan2(-det_product, dot_product) * 180 / np.pi
            if angle_full < 0:
                return 360 + angle_full
            else:
                return angle_full
        else:
            return np.arccos(dot_product) * 180 / np.pi

    def calc_angle_edges(self, edge_1, edge_2, angle_360=False):
        """
        Computes the angle between any two edges
        :param edge_1: first edge (id, not (x,y)-coordinates!
        :param edge_2: second edge (id, not (x,y)-coordinates!
        :param angle_360: boolean for returning angle in 360 or 180 180 degrees spectrum
        :return:
        """
        common_vertex = np.intersect1d(self.edge_to_vertex[edge_1], self.edge_to_vertex[edge_2])
        other_vertex_1 = self.edge_to_vertex[edge_1, self.edge_to_vertex[edge_1] != common_vertex]
        other_vertex_2 = self.edge_to_vertex[edge_2, self.edge_to_vertex[edge_2] != common_vertex]
        vec_m = self.vertices[common_vertex].flatten() - self.vertices[other_vertex_1].flatten()
        vec_p = self.vertices[common_vertex].flatten()  - self.vertices[other_vertex_2].flatten()
        return self.calc_angles_vectors_2D(vec_m, vec_p, angle_360=angle_360)

    def check_angles_leaving_vertices(self, vertex_id):
        """
        Calculates all the angles between all vertices leaving vertex_id
        :param vertex_id: integer with id for vertex
        :return:
        """
        edges = np.array(self.vertex_to_edge[vertex_id])
        num_edges = len(edges)
        if num_edges == 2:
            return self.calc_angle_edges(edges[0], edges[1]), np.array([[edges[0], edges[1]]])
        elif num_edges == 1:
            return 360, None

        angles = np.zeros((num_edges,))

        # Determine order, starting with 0 angle at positive y-axis [0, 1]:
        angles_y_axis = np.zeros((num_edges,))
        vec_p = np.array([1, 0])
        for ii, edge in enumerate(edges):
            other_vertex = self.edge_to_vertex[edge][self.edge_to_vertex[edge] != vertex_id]
            vec_m = self.vertices[other_vertex].flatten() - self.vertices[vertex_id].flatten()
            angles_y_axis[ii] = self.calc_angles_vectors_2D(vec_m, vec_p, angle_360=True)

        sort_edges_ids = edges[np.argsort(angles_y_axis)]
        edge_pair = np.zeros((num_edges, 2), dtype=int)
        for ii in range(num_edges - 1):
            edge_pair[ii] = [sort_edges_ids[ii + 1], sort_edges_ids[ii]]
            angles[ii] = self.calc_angle_edges(sort_edges_ids[ii + 1], sort_edges_ids[ii], angle_360=True)
        edge_pair[-1] = [sort_edges_ids[0], sort_edges_ids[-1]]
        angles[-1] = self.calc_angle_edges(sort_edges_ids[0], sort_edges_ids[-1], angle_360=True)

        if abs(np.sum(angles) - 360) > 1e-4:
            print('Mmmhhh...')
        return angles, edge_pair

    def straighten_edges(self, tolerance_angle, char_len, correct_aperture):
        """
        Method which straightens fractures if within certain tolerance from a straight line
        :param tolerance_angle: deviation from straight line
        :param char_len: radius of algebraic constraint
        :param correct_aperture: boolean if applying aperture correction
        :return:
        """
        for ii in range(self.get_num_vertices()):
            if not self.active_vertices[ii]:
                continue
            edges = self.vertex_to_edge[ii]
            if len(edges) == 2:
                angle = self.calc_angle_edges(edges[0], edges[1])

                if np.abs(angle - 180) <= tolerance_angle:
                    # Edge straight enough to be merged into closest vertex:
                    other_vertices = self.edge_to_vertex[edges].flatten()[np.where(self.edge_to_vertex[edges].flatten() != ii)[0]]
                    dist_vertices = np.linalg.norm(self.vertices[ii] - self.vertices[other_vertices], axis=1)
                    closests_id = np.argmin(dist_vertices)
                    self.merge_vertices(vertex_id_from=ii, vertex_id_to=other_vertices[closests_id], char_len=char_len, correct_aperture=correct_aperture)
        return 0

    def remove_small_angles(self, tolerange_small_angle, char_len, correct_aperture):
        """
        Method which removes small angles which might result in "skinny" triangles in meshed results
        :param tolerange_small_angle: max. allowable angle between two fracture segments
        :param char_len: radius of algebraic constraint
        :param correct_aperture: boolean if applying aperture correction
        :return:
        """
        active_ids = np.where(self.active_vertices)[0]

        for vertex_id in active_ids:
            if not self.active_vertices[vertex_id]:
                continue

            angles, edge_ids = self.check_angles_leaving_vertices(vertex_id)
            while np.any(angles < tolerange_small_angle):
                # Determine which angle is the smallest and merge two edges into one, logic: keep largest edge and merge
                # non-shared vertex into closest vertex, then update angles and continue
                edge_pair = edge_ids[np.argmin(angles)]
                other_vertices = self.edge_to_vertex[edge_pair].flatten()[self.edge_to_vertex[edge_pair].flatten() != vertex_id]
                length_vec = np.linalg.norm(self.vertices[vertex_id] - self.vertices[other_vertices], axis=1)
                vertex_from = other_vertices[np.argmin(length_vec)]
                large_edge = edge_pair[np.argmax(length_vec)]

                dist_vec = np.linalg.norm(self.vertices[vertex_from] - self.vertices[self.edge_to_vertex[large_edge]], axis=1)
                vertex_to = self.edge_to_vertex[large_edge][np.argmin(dist_vec)]
                self.merge_vertices(vertex_from, vertex_to, char_len, correct_aperture)
                angles, edge_ids = self.check_angles_leaving_vertices(vertex_id)
        return 0

    def simplify_graph(self, order_discr, char_len, merge_treshold=0.66,
                       tolerange_small_angle=20, small_angle_iter=2, tolerange_straight_angle=7.5, correct_aperture=True, straighten_edges=False):
        """
        Method which performs actual simplification of the graph
        :param order_discr: order in which vertices are sequentially checked on algebraic constraint
        :param char_len: radius of algebraic constraint
        :param merge_treshold: h-factor which is recommended between [0.5, 0.86]
        :param tolerange_small_angle: max. allowable angle between two fracture segments
        :param small_angle_iter: number of times to run small angle correction, merging vertices changes angle and might
                introduce new accute intersection
        :param tolerange_straight_angle: allowable deviation from straight line
        :param correct_aperture: boolean to applying aperture correction or not
        :param straighten_edges: boolean to straighten edges (calling method which does this) after preprocessing to
                speed up succesive gridding
        :return:
        """
        assert 0.5 <= merge_treshold <= 0.86, "Choose threshold on closed interval [0.5, 0.86]"
        self.closest_point_method(order_discr, char_len, merge_treshold, correct_aperture)
        for ii in range(small_angle_iter):
            self.remove_small_angles(tolerange_small_angle, char_len, correct_aperture)
        if straighten_edges:
            self.straighten_edges(tolerange_straight_angle, char_len, correct_aperture)
        return 0

    def visualize_edges(self, vertex_id):
        """
        Method which plots fracture network around vertex_id
        :param vertex_id: integer id for current vertex
        :return:
        """
        edges = self.vertex_to_edge[vertex_id]
        leaving_vertices = self.edge_to_vertex[edges].flatten()[np.where(self.edge_to_vertex[edges].flatten() != vertex_id)[0]]
        plt.figure()
        plt.plot(
            np.vstack((self.vertices[self.edge_to_vertex[edges, 0], 0].T,
                       self.vertices[self.edge_to_vertex[edges, 1], 0].T)),
            np.vstack((self.vertices[self.edge_to_vertex[edges, 0], 1].T,
                       self.vertices[self.edge_to_vertex[edges, 1], 1].T)),
            color='black'
        )
        plt.axis('equal')
        plt.plot(self.vertices[leaving_vertices, 0], self.vertices[leaving_vertices, 1], '.', color='red')
        plt.plot(self.vertices[vertex_id, 0], self.vertices[vertex_id, 1], '.', color='blue')
        plt.show()

    def visualize_graph(self):
        """
        Method which visualize current graph state
        :return:
        """
        num_edges = self.get_num_edges()
        num_vertices = self.get_num_vertices()
        plt.figure()
        plt.plot(
            np.vstack((self.vertices[self.edge_to_vertex[:num_edges, 0], 0].T,
                       self.vertices[self.edge_to_vertex[:num_edges, 1], 0].T)),
            np.vstack((self.vertices[self.edge_to_vertex[:num_edges, 0], 1].T,
                       self.vertices[self.edge_to_vertex[:num_edges, 1], 1].T)),
            color='black'
        )
        plt.axis('equal')
        plt.plot(self.vertices[:num_vertices, 0], self.vertices[:num_vertices, 1], '.', color='red')
        plt.show()
        return 0

    def visualize_sub_graph(self, edges):
        """
        Visualize subgraph based on list of edges
        :param edges: subset of self.edge_to_vertex containing vertex pairs of sub-graph
        :return:
        """
        vertices = list(set(edges.flatten()))
        epsilon = 0.1
        plt.figure()
        for edge in edges:
            plt.plot(
                np.vstack((self.vertices[edge[0], 0].T,
                           self.vertices[edge[1], 0].T)),
                np.vstack((self.vertices[edge[0], 1].T,
                           self.vertices[edge[1], 1].T)),
                color='black'
            )
            plt.text((self.vertices[edge[0], 0] + self.vertices[edge[1], 0]) / 2 + epsilon,
                     (self.vertices[edge[0], 1] + self.vertices[edge[1], 1]) / 2 + epsilon,
                     str(np.intersect1d(self.vertex_to_edge[edge[0]], self.vertex_to_edge[edge[1]])[0]), fontsize=10)
        plt.axis('equal')
        for ii in vertices:
            plt.plot(self.vertices[ii, 0], self.vertices[ii, 1], '.', color='red')
            plt.text(self.vertices[ii, 0] + epsilon, self.vertices[ii, 1] + epsilon,
                     str(ii), fontsize=10)
        plt.show()
        return 0

    def check_collapsed_edges(self):
        """
        Safety method to check graph consistency on collapsed edges that might occur in domain
        :return:
        """
        check_result = False

        active_edge_ids = np.where(self.active_edges)[0]
        for edge in active_edge_ids:
            vertices = self.edge_to_vertex[edge]
            if vertices[0] == vertices[1]:
                check_result = True
                break
        return check_result

    def check_duplicate_edges(self):
        """
        Safety method to check graph consistency on duplicate edges that might occur in domain
        :return:
        """
        check_result = False
        slice_edge_to_vertex = self.edge_to_vertex[0:self.get_num_edges()]
        unique_edges = np.unique(slice_edge_to_vertex, axis=0)
        if unique_edges.shape[0] != slice_edge_to_vertex.shape[0]:
            check_result = True
        return check_result

    def check_graph_consistency(self):
        """
        Check consistency of the graph
        :return:
        """
        self.check_collapsed_edges()
        self.check_duplicate_edges()
        return 0

    def connectivity_merging_vertices(self, vertex_from, vertex_to, radius):
        """
        Compute connectivity between vertex_from and vertex_to using a sub-graph and Dijkstra's shortest path
        :param vertex_from: id for vertex which is merged
        :param vertex_to: id for vertex which is fixed
        :param radius: radius around which vertices and edges are extract for the subgraph, choosing this parameter too
                large can result in unexpected results or long preprocessing time
        :return: resistance (L / aperture^2), is infinity if no direct connection exists
        """
        # Check if nodes are first-degree connected
        common_edge = np.intersect1d(self.vertex_to_edge[vertex_from], self.vertex_to_edge[vertex_to])
        if common_edge.size != 0:
            return np.linalg.norm(self.vertices[vertex_from].flatten() - self.vertices[vertex_to].flatten()) / (self.apertures[common_edge[0]] ** 2)
        else:
            # Extract subpgrah of vertices and edges and their weights (within a radius of char_len * X around vertices)
            vertices_in_radius = np.where(np.linalg.norm(self.vertices[:self.get_num_vertices()] -
                                                         self.vertices[vertex_from], axis=1) < radius)[0]

            edges_in_radius = []
            for ii in vertices_in_radius:
                edges_in_radius += self.vertex_to_edge[ii]
            edges_in_radius = list(set(edges_in_radius))
            vertices_in_radius = list(set(self.edge_to_vertex[edges_in_radius].flatten()))
            if vertex_to not in vertices_in_radius:
                return np.inf

            edges = self.edge_to_vertex[edges_in_radius]
            edge_weights = np.linalg.norm(self.vertices[edges[:, 0]] - self.vertices[edges[:, 1]], axis=1) / \
                           self.apertures[edges_in_radius] ** 2 # m to km and m for aperture to cm (note: ** 2) in order to avoid extremely large values

            # Construct subgraph with weights in igraph and calculate shortest_path between vertices
            g = igraph.Graph(edges=edges, edge_attrs={'weight': edge_weights})
            dist_from_to = g.shortest_paths(source=vertex_from, target=vertex_to, weights=edge_weights)
            return dist_from_to[0][0]

    def status_edge_after_merge(self, vertex_from, vertex_to, edge):
        """
        Determines status of edge leaving vertex_from when applying merge into vertex_to
        :param vertex_from: vertex id which is merged
        :param vertex_to: vertex id which remains in domain
        :param edge: id of edge which is leaving vertex_from
        :return: status of edge (str), list of two vertices making up new edge, id of new edge (or old, if simply
                extended)
        """
        if np.all(np.sort(self.edge_to_vertex[edge]) == np.sort(np.array([vertex_from, vertex_to]))):
            # Edge will collapse:
            return 'collapsed', np.array([vertex_to, vertex_to]), -1
        else:
            # Loop over all edges in vertex_to and check of the new edge will overlap
            new_edge = np.sort(np.array([vertex_to, self.edge_to_vertex[edge, self.edge_to_vertex[edge] != vertex_from][0]], dtype=int))
            overlap = False
            new_edge_id = -1
            for curr_edge in self.vertex_to_edge[vertex_to]:
                if np.all(np.sort(self.edge_to_vertex[curr_edge]) == new_edge):
                    overlap = True
                    new_edge_id = curr_edge
            if overlap:
                return 'overlap', new_edge, new_edge_id
            else:
                return 'extension', new_edge, edge

    def calc_effective_aperture_and_heat_transfer_sequential(self, new_edge, collapsed_edge_id, extended_edge_id, resistance, vertex_from, vertex_to):
        """
        Calculate effect aperture and heat transfer for Type 2 corrections (similar to sequential resistors in circuit)
        :param new_edge: list of pair of vertices making up new edge
        :param collapsed_edge_id: list with edge id if edge of other edge is it collapsed (is empty if no edge connects
                to this vertex_from
        :param extended_edge_id: id of the edge that is currently evaluated (
        :param resistance: L / aperture^2 , computed using Dijkstra's shortest path (== inf if it doesn't exist)
        :param vertex_from: id vertex which is merged
        :param vertex_to: id vertex which stays in domain
        :return: effective_aperture, effective_heattransfer
        """
        len_new_edge = np.linalg.norm(self.vertices[new_edge[0]].flatten() - self.vertices[new_edge[1]].flatten())
        len_old_edge = np.linalg.norm(self.vertices[self.edge_to_vertex[extended_edge_id, 0]].flatten() -
                                      self.vertices[self.edge_to_vertex[extended_edge_id, 1]].flatten())

        dist_gap = np.linalg.norm(self.vertices[vertex_from].flatten() - self.vertices[vertex_to].flatten())
        weights = np.array([len_old_edge, dist_gap]) / (len_old_edge + dist_gap)
        n_pow = 9.5608 * weights[1] + 1.18024  # fitted from tested different gaps and different lc!!!
        if resistance == np.inf:
            if self.matrix_eff_aperture < self.apertures[extended_edge_id]:
                eff_aperture = 1 / ((1 - weights[1] ** n_pow) / (self.apertures[extended_edge_id]) + (weights[1] ** n_pow) / (self.matrix_eff_aperture))
            else:
                weights = np.array([len_old_edge, dist_gap]) / (len_old_edge + dist_gap)
                eff_aperture = (self.apertures[extended_edge_id] * weights[0]) + (self.matrix_eff_aperture * weights[1])
            eff_heat_transfer = 1

        else:
            dist_gap = np.max((len_new_edge - len_old_edge, 0.1))
            weights = np.array([len_old_edge, dist_gap]) / (len_old_edge + dist_gap)
            eff_gap_aper = np.sqrt(dist_gap / resistance)
            if eff_gap_aper > np.max(self.apertures):
                eff_gap_aper = np.max(self.apertures)
            eff_aperture = 1 / (weights[0] / (self.apertures[extended_edge_id]) + weights[1] / eff_gap_aper)
            if collapsed_edge_id.size == 0:
                len_other_edge = 0
            else:
                len_other_edge = np.linalg.norm(self.vertices[self.edge_to_vertex[collapsed_edge_id[0], 0]].flatten() -
                                                self.vertices[self.edge_to_vertex[collapsed_edge_id[0], 1]].flatten())
            eff_heat_transfer = (len_other_edge + len_old_edge) / len_new_edge

        return eff_aperture, eff_heat_transfer

    def calc_effective_aperture_and_heat_transfer_parallel(self, old_edge_id, new_edge_id):
        """
        Simple arithmetic mean weighted by length of the two edges which are overlapping after merge (similar to
                parallel resistors in circuit)
        :param old_edge_id: id of the old edge after merging
        :param new_edge_id: id of the new edge after merging
        :return: effective_aperture, effective_heattransfer
        """
        len_new_edge = np.linalg.norm(self.vertices[self.edge_to_vertex[new_edge_id, 0]].flatten() -
                                      self.vertices[self.edge_to_vertex[new_edge_id, 1]].flatten())
        len_old_edge = np.linalg.norm(self.vertices[self.edge_to_vertex[old_edge_id, 0]].flatten() -
                                      self.vertices[self.edge_to_vertex[old_edge_id, 1]].flatten())
        eff_aperture = np.sqrt(len_new_edge * (self.apertures[new_edge_id] ** 2 / len_new_edge +
                                       self.apertures[old_edge_id] ** 2 / len_old_edge))
        eff_heat_transfer = (len_new_edge + len_old_edge) / len_new_edge
        return eff_aperture, eff_heat_transfer

    def update_volume_collapsed_edge(self, vertex_from, vertex_to, edge_id):
        """
        Updates volumes after an edge is collapsed, basically distributes the volumes to all connecting edges to
                vertex_from
        :param vertex_from: id vertex which is merged
        :param vertex_to: id vertex which remains in domain
        :param edge_id: id of edge currently under consideration
        :return:
        """
        # Find if neighbouring vertices have edges, if both have then distribute 50% to both vertices and edges respectively
        edges_vertex_from = copy.deepcopy(self.vertex_to_edge[vertex_from])
        edges_vertex_to = copy.deepcopy(self.vertex_to_edge[vertex_to])
        volume_collapsed_edge = copy.deepcopy(self.volumes[edge_id])
        if edge_id in edges_vertex_from and edge_id in edges_vertex_to:
            other_edges_vertex_from = edges_vertex_from
            other_edges_vertex_from.remove(edge_id)
            for edge in other_edges_vertex_from:
                self.volumes[edge] += volume_collapsed_edge / len(other_edges_vertex_from) / 2

            other_edges_vertex_to = edges_vertex_to
            other_edges_vertex_to.remove(edge_id)
            for edge in other_edges_vertex_to:
                self.volumes[edge] += volume_collapsed_edge / len(other_edges_vertex_to) / 2

        elif edge_id in edges_vertex_from:
            other_edges_vertex_from = edges_vertex_from
            other_edges_vertex_from.remove(edge_id)
            for edge in other_edges_vertex_from:
                self.volumes[edge] += volume_collapsed_edge / len(other_edges_vertex_from)

        elif edge_id in edges_vertex_to:
            other_edges_vertex_to = edges_vertex_to
            other_edges_vertex_to.remove(edge_id)
            for edge in other_edges_vertex_to:
                self.volumes[edge] += volume_collapsed_edge / len(other_edges_vertex_to)
        self.volumes[edge_id] = 0
        return 0

    def update_volume_overlap_edge(self, old_edge_id, new_edge_id):
        """
        Update volume of overlapping and removed edge
        :param old_edge_id: id of edge which is removed after merging
        :param new_edge_id: id of edge which remains in domain after merging
        :return:
        """
        # Add volume old edge to new edge
        self.volumes[new_edge_id] += copy.deepcopy(self.volumes[old_edge_id])
        self.volumes[old_edge_id] = 0
        return 0

    def update_volumes_and_apertures(self, vertex_from, vertex_to, status_after_merging,
                                     edge_to_vertex_after_merging, edge_id_after_merge, char_len):
        """
        Main function that updates volumes and apertures after merging
        :param vertex_from: id vertex which gets merged
        :param vertex_to: id vertex which remains in domain (target vertex)
        :param status_after_merging: status of edges connecting to vertex_from after merging
        :param edge_to_vertex_after_merging: pair of vertices of edges leaving vertex_from after merging
        :param edge_id_after_merge: ids of edges leaving vertex_from after merging
        :param char_len: radius of algebraic constraint
        :return:
        """
        # First check if any of the edges leaving vertex_from is collapsing, distribute volume first, then loop
        # over the remaining edges to update properties accordingly
        resistance = self.connectivity_merging_vertices(vertex_from, vertex_to, char_len * 2.5)
        if 'collapsed' in status_after_merging:
            self.update_volume_collapsed_edge(vertex_from, vertex_to,
                                              self.vertex_to_edge[vertex_from][status_after_merging.index('collapsed')])

        for ii, status_edge in enumerate(status_after_merging):
            if status_edge == 'collapsed':
                continue
            elif status_edge == 'overlap':
                # Parallel:
                old_edge_id = self.vertex_to_edge[vertex_from][ii]
                new_edge_id = edge_id_after_merge[ii]
                eff_aperture, eff_heat_transfer = self.calc_effective_aperture_and_heat_transfer_parallel(old_edge_id, new_edge_id)
                self.apertures[new_edge_id] = eff_aperture
                self.heat_transfer_mult[new_edge_id] = eff_heat_transfer
                self.update_volume_overlap_edge(old_edge_id, new_edge_id)
            elif status_edge == 'extension':
                # Sequential:
                new_edge = edge_to_vertex_after_merging[ii]
                collapsed_edge_id = np.intersect1d(self.vertex_to_edge[vertex_from], self.vertex_to_edge[vertex_to])
                extended_edge_id = self.vertex_to_edge[vertex_from][ii]
                eff_aperture, eff_heat_transfer = self.calc_effective_aperture_and_heat_transfer_sequential(
                    new_edge, collapsed_edge_id, extended_edge_id, resistance, vertex_from, vertex_to)
                self.apertures[extended_edge_id] = eff_aperture
                self.heat_transfer_mult[extended_edge_id] = eff_heat_transfer
        # self.visualize_sub_graph(self.edge_to_vertex[list(set(self.vertex_to_edge[vertex_from] + self.vertex_to_edge[vertex_to]))])
        return 0

    def visualize_graph_with_volume_weights(self, min_val=None, max_val=None):
        """
        Visualze edges in graph by volume
        :param min_val: minimum volume for scaling colorbar and data
        :param max_val: maximum volume for scaling colorbar and data
        :return:
        """
        fracs = copy.deepcopy(self.volumes[self.active_edges])
        if min_val is None:
            min_val = np.min(fracs)

        if max_val is None:
            max_val = np.min(fracs)

        fracs[fracs < min_val] = min_val
        fracs[fracs > max_val] = max_val
        norm = colors.Normalize(min_val, max_val)
        colors_aper = cm.viridis(norm(fracs))

        plt.figure()
        for jj in range(self.get_num_edges()):
            if not self.active_edges[jj]:
                continue
            plt.plot(
                np.vstack((self.vertices[self.edge_to_vertex[jj, 0], 0].T,
                           self.vertices[self.edge_to_vertex[jj, 1], 0].T)),
                np.vstack((self.vertices[self.edge_to_vertex[jj, 0], 1].T,
                           self.vertices[self.edge_to_vertex[jj, 1], 1].T)),
                color=colors_aper[jj, :-1]
            )
        plt.axis('equal')
        plt.title('Volume Weights')
        plt.show()
        return

    def visualize_graph_with_aperture_weights(self, min_val=None, max_val=None):
        """
        Visualze edges in graph by aperture
        :param min_val: minimum aperture for scaling colorbar and data
        :param max_val: maximum aperture for scaling colorbar and data
        :return:
        """
        fracs = copy.deepcopy(self.apertures[self.active_edges])
        if min_val is None:
            min_val = np.min(fracs)

        if max_val is None:
            max_val = np.min(fracs)

        fracs[fracs < min_val] = min_val
        fracs[fracs > max_val] = max_val
        norm = colors.Normalize(min_val, max_val)
        colors_aper = cm.viridis(norm(fracs))

        plt.figure()
        for jj in range(self.get_num_edges()):
            if not self.active_edges[jj]:
                continue
            plt.plot(
                np.vstack((self.vertices[self.edge_to_vertex[jj, 0], 0].T,
                           self.vertices[self.edge_to_vertex[jj, 1], 0].T)),
                np.vstack((self.vertices[self.edge_to_vertex[jj, 0], 1].T,
                           self.vertices[self.edge_to_vertex[jj, 1], 1].T)),
                color=colors_aper[jj, :-1]
            )
        plt.axis('equal')
        plt.title('Aperture Weights')
        plt.show()
        return


def create_geo_file(act_frac_sys, filename, decimals,
                    height_res, char_len, box_data, char_len_boundary, export_frac=True):
    """
    Creates geo file which serves as input to gmsh
    :param act_frac_sys: list of fractures in domain in format [[x1, y1, x2, y2], [...], ..., [...]]
    :param filename: name of the resulting geo-file
    :param decimals: data is rounded off to this number of decimals
    :param height_res: height of the resulting 1-layer 3D reservoir
    :param char_len: characteristic length of the resulting mesh
    :param box_data: coordinates of the box-data around the fracture network
    :param char_len_boundary: characteristic length of mesh elements at the boundary
    :param export_frac: boolean which exports fractures into the meshed file
    :return:
    """
    act_frac_sys = np.round(act_frac_sys * 10 ** decimals) * 10 ** (-decimals)
    num_segm_tot = act_frac_sys.shape[0]
    unique_nodes = np.unique(np.vstack((act_frac_sys[:, :2], act_frac_sys[:, 2:])), axis=0)
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
    points_created = np.zeros((num_nodes_tot,), dtype=bool)
    line_count = 0
    point_count = 0

    for ii in act_frac_sys:
        # Take two points per segment and write to .geo file:
        # e.g. point: Point(1) = {.1, 0, 0, lc};
        nodes = np.zeros((2,), dtype=int)
        nodes[0] = np.where(np.logical_and(ii[0] == unique_nodes[:, 0], ii[1] == unique_nodes[:, 1]))[0]
        nodes[1] = np.where(np.logical_and(ii[2] == unique_nodes[:, 0], ii[3] == unique_nodes[:, 1]))[0]

        # Check if first point is already created, if not, add it:
        if not points_created[nodes[0]]:
            points_created[nodes[0]] = True
            point_count += 1
            f.write('Point({:d}) = {{{:8.5f}, {:8.5f}, 0, lc}};\n'.format(nodes[0] + 1,
                                                                           unique_nodes[nodes[0], 0],
                                                                           unique_nodes[nodes[0], 1]))

        if not points_created[nodes[1]]:
            points_created[nodes[1]] = True
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
    # f.write('Extrude {0, 0, height_res}{ Line {1:num_lines_frac}; Layers{1}; Recombine;}\n')
    f.write('num_surfaces_after = news - 1;\n')
    f.write('num_surfaces_fracs = num_surfaces_after - num_surfaces_before;\n\n')
    for ii in range(act_frac_sys.shape[0]):
        # f.write('Physical Surface({:d}) = {{num_surfaces_before + {:d}}};\n'.format(90000 + ii, ii))
        f.write('Extrude {{0, 0, height_res}}{{ Line {{{:d}}}; Layers{{1}}; Recombine;}}\n'.format(ii + 1))
        if export_frac:
            f.write('Physical Surface({:d}) = {{news - 1}};\n'.format(90000 + ii))

    # Create mesh and perform coherency check:
    f.write('Mesh 3;  // Generalte 3D mesh\n')
    f.write('Coherence Mesh;  // Remove duplicate entities\n')
    f.write('Mesh.MshFileVersion = 2.1;\n')
    f.close()
    return 0
