"""
Module that calculates the intersection point of two given line segments
Rewritten from a matlab script by Stephan de Hoop

Author: Joey Herbold
Date: 22/11/2020
"""

import numpy as np


def find_parametric_intersect(ii_frac, jj_frac):
    """
    This function calculates the intersection point of two fractures

    :param ii_frac: Fracture 1
    :param jj_frac: Fracture 2
    :return t: Parametric distance along fracture 1 value between 0 and 1
    :return s: Parametric distance along fracture 2 value between 0 and 1
    :return int_coord: Coordinates of the intersection
    """
    P_0 = np.array([ii_frac[0],
                    ii_frac[1]])

    P = np.array([ii_frac[2] - ii_frac[0],
                  ii_frac[3] - ii_frac[1]])

    Q_0 = np.array([jj_frac[0],
                    jj_frac[1]])

    Q = np.array([jj_frac[2] - jj_frac[0],
                  jj_frac[3] - jj_frac[1]])

    A = np.zeros((2, 2))
    A[:, 0] = P
    A[:, 1] = -Q

    if abs(np.linalg.det(A)) < 1e-16:
        # Check if the lines are parallel by looking at the determinant
        t = -1
        s = -1
        int_coord = np.array([])
    else:
        # Calculate the intersection point of the two lines
        rhs = P_0 - Q_0

        solution = np.linalg.solve(-A, rhs)

        t = solution[0]
        s = solution[1]

        int_coord = P_0 + t * P

    return t, s, int_coord
