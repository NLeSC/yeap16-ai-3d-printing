#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

"""Utility functions"""


def block_index(num_blocks, len_list):
    """
    Generate a list with equally space numbers from 0 to num_blocks.

    INPUT   (1) int 'num_blocks': Number of blocks
    OUTPUT  (1) list
    """
    if num_blocks < 1:
        ValueError('Number of blocks must be larger or equal to 1')

    if len_list < num_blocks:
        ValueError('Length of list must be larger than number of blocks.')

    # Generate linear list from 0 to num_blocks
    lin_list = np.linspace(0, num_blocks, len_list, endpoint=False)

    # Return floored linear list as integers
    return np.floor(lin_list).astype('uint8')


def limit_mem():
    """Free up memory between GPU processes."""
    # Clear memory of alternative TensforFlow session
    K.get_session().close()

    # Allow current Session to grow with freed memory
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def get_closest_factors(number):
    """Find the 2 factors of a number that are closest together."""
    # Find first factor of number of checking for integer root
    a = int(np.sqrt(number))
    while number % a != 0:
        a -= 1
    b = number/a

    # If smallest factor is 1, recurse with incremented number
    if a == 1 or b == 1:
        a, b = get_closest_factors(number + 1)

    # Return factors
    return a, b
