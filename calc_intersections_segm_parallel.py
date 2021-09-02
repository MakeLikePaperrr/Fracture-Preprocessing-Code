from calc_intersections_segm import calc_intersections_segm
from partition_domain import partition_domain
from merge_domain import merge_domain

import numpy as np
import multiprocessing as mp


def calc_intersections_worker(partition, send_end, act_frac_sys, frac_order_vec, tolerance_intersect, tolerance_zero):
    act_frac_sys, frac_order_vec = calc_intersections_segm(act_frac_sys, frac_order_vec, tolerance_intersect, tolerance_zero)
    send_end.send((act_frac_sys, frac_order_vec))


def calc_intersections_segm_parallel(act_frac_sys, frac_order_vec, tolerance_intersect, tolerance_zero, number_partitions_x, number_partitions_y):
    """

    :param act_frac_sys:
    :param frac_set_vec:
    :param tolerance_intersect:
    :param tolerance_zero:
    :param number_partitions_x:
    :param number_partitions_y:
    :return:
    """
    number_partitions = number_partitions_x * number_partitions_y

    if number_partitions > 1:
        act_frac_sys_list, frac_order_vec_list, partition_lines = partition_domain(act_frac_sys, frac_order_vec,
                                                                                                      tolerance_intersect, number_partitions_x,
                                                                                                      number_partitions_y)

        jobs = []
        pipe_list = []
        for partition in range(number_partitions):
            recv_end, send_end = mp.Pipe(False)
            p = mp.Process(target=calc_intersections_worker, args=(partition,
                                                                   send_end,
                                                                   act_frac_sys_list[partition],
                                                                   frac_order_vec_list[partition],
                                                                   tolerance_intersect,
                                                                   tolerance_zero,))
            jobs.append(p)
            pipe_list.append(recv_end)
            p.start()

        act_frac_sys_list = []
        frac_order_vec_list = []
        for p in pipe_list:
            recv = p.recv()
            act_frac_sys_list.append(recv[0])
            frac_order_vec_list.append(recv[1])

        for p in jobs:
            p.join()

        for p in jobs:
            p.terminate()

        return act_frac_sys_list, frac_order_vec_list, partition_lines
    else:
        act_frac_sys, frac_order_vec = calc_intersections_segm(act_frac_sys, frac_order_vec, tolerance_intersect, tolerance_zero)
        return [act_frac_sys], [frac_order_vec], []