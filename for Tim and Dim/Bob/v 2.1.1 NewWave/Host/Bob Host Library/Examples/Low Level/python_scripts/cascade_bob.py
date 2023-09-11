import numpy as np
from typing import Union
import math
import numpy_indexed as npi
import collections
import bitstring
import os
import socket
import logging as log

# TODO сделать сначала test connections, научиться нормально передавать np.array по сокету

# logging config
log.basicConfig(level=log.DEBUG, filename='alice_log.log', filemode='w',
                format='%(asctime)s - %(levelname)s - %(message)s')

# cascade config
SEED = 1337
MAX_ITER = 4
MAGIC_CONSTANT = 0.73
MAX_KEY_LENGTH_BYTES = 10000

# network config
ENCODING = 'utf-8'
CONNECTION_TIMEOUT = 20
DISCONNECT_MSG = '!DISCONNECT'


# returns raw key length, number of corrected bit errors
def cascade_bob(qber, bob_key_path, alice_ip='10.32.64.33', alice_port=9090):
    # cascade initialization
    bit_errors_corrected = 0
    np.random.seed(SEED)

    with open(bob_key_path, 'rb') as file:
        hex = file.read(MAX_KEY_LENGTH_BYTES).hex()
        bitarray = bitstring.BitArray(hex=hex).bin
        key = np.fromiter(bitarray, dtype=np.int8)

    log.info(f'key length: {key.shape[0]}, nbytes: {key.nbytes}')

    n = key.shape[0]

    permutations = [np.random.permutation(n) for i in range(MAX_ITER)]

    l = [int(np.round(MAGIC_CONSTANT / qber)) * 2 ** i for i in range(MAX_ITER)]  # qber in decimals not in %

    # trimming the key
    n = n - n % l[-1]
    key = key[:n]

    # network initialization
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(CONNECTION_TIMEOUT)

    # blocking
    client.connect((alice_ip, alice_port))

    log.info(f'connected to server: {alice_ip}')

    # starting cascade
    for i in range(MAX_ITER):
        permuted_key = key[permutations[i]]
        num_blocks = n // l[i]
        start_indexes = np.arange(0, n, l[i], dtype=int)

        bob_parity_vector = get_parity_vector(permuted_key, num_blocks, start_indexes, start_indexes + l[i])

        # todo научиться передавать np.array
        client.sendall(bob_parity_vector.tobytes())

        alice_parity_vector = recv_alices_parity_vector(client, num_blocks)

        blocks_with_errors = np.logical_xor(alice_parity_vector, bob_parity_vector)

        # batch binary
        error_indexes = batch_binary(client, permuted_key, l[i],
                                     start_indexes[blocks_with_errors])  # error indexes relative to permuted_key[0]

        # correct errors in bob's key
        original_error_indexes = permutations[i][error_indexes]
        key[original_error_indexes] = np.logical_xor(key[original_error_indexes], 1)
        bit_errors_corrected += error_indexes.shape[0]

        # lookback
        if i > 1 and error_indexes.shape[0] != 0:
            bit_errors_corrected += lookback(client, key, n, permutations, l, current_lvl=i,
                                             original_error_indexes=original_error_indexes)

    # todo write to file on bob

    # destruction
    client.close()
    log.info(f'disconnected from server: {alice_ip}')

    return key.shape[0], bit_errors_corrected


def filter_odd_frequency(array: np.array) -> list:
    """
    array: arbitrary np.array of integers

    return: all values with odd number of occurences
    """
    freq = collections.Counter(array)  # get frequencies
    odd_freq = [k for k, v in freq.items() if v % 2 == 1]  # filter keys with odd values
    return odd_freq


def lookback(client: socket, key: np.array, n: int, permutations: list, l: list, current_lvl: int,
             original_error_indexes: np.array):
    bit_errors_corrected = 0

    block_contains_error = [np.zeros(n // l[i], dtype=bool) for i in range(current_lvl + 1)]

    # initialize block_contains_error
    for j in range(current_lvl):
        permuted_indexes = npi.indices(permutations[j], original_error_indexes)
        permuted_blocks_with_errors = filter_odd_frequency(permuted_indexes // l[j])
        block_contains_error[j][permuted_blocks_with_errors] = 1

    while True:
        errors = np.array([np.sum(block_contains_error[i]) for i in range(current_lvl + 1)])

        if np.sum(errors) == 0:
            break

        max_errors_lvl = np.argmax(errors)

        start_indexes = np.arange(0, n, l[max_errors_lvl], dtype=int)
        error_indexes = batch_binary(client,
                                     key[permutations[max_errors_lvl]],
                                     l[max_errors_lvl],
                                     start_indexes[block_contains_error[max_errors_lvl]])

        # correct errors in bob's key
        original_error_indexes = permutations[max_errors_lvl][error_indexes]
        key[original_error_indexes] = np.logical_xor(key[original_error_indexes], 1)
        bit_errors_corrected += error_indexes.shape[0]

        for j in range(current_lvl + 1):
            permuted_indexes = npi.indices(permutations[j], original_error_indexes)
            permuted_blocks_with_errors = filter_odd_frequency(permuted_indexes // l[j])
            block_contains_error[j][permuted_blocks_with_errors] = np.logical_xor(
                block_contains_error[j][permuted_blocks_with_errors], 1)

        # sanity check
        if not np.all(block_contains_error[max_errors_lvl] == 0):
            # log.debug(f'not all errors found at max_errors_lvl after running batch binary')
            raise RuntimeError('not all errors found at max_errors_lvl after running batch binary')

    return bit_errors_corrected


def batch_binary(client: socket, permuted_key: np.array, l: int, start_indexes: np.array) -> np.array:
    if start_indexes.shape[0] == 0:
        log.debug(f'empty start indexes in batch binary')
        return start_indexes

    left_ptr = start_indexes
    right_ptr = start_indexes + l - 1
    num_blocks = left_ptr.shape[0]
    depth = math.ceil(math.log2(l))
    for j in range(depth):
        middle_ptr = np.floor((left_ptr + right_ptr) / 2).astype(int)

        bob_left_parity = get_parity_vector(permuted_key, num_blocks, left_ptr, middle_ptr + 1)

        # todo научиться передавать np.array
        client.sendall(bob_left_parity.tobytes())

        alice_left_parity = recv_alices_parity_vector(client, num_blocks)

        do_step_right = np.invert(np.logical_xor(alice_left_parity, bob_left_parity))
        left_ptr = np.invert(do_step_right) * left_ptr + do_step_right * (middle_ptr + 1)
        right_ptr = do_step_right * right_ptr + np.invert(do_step_right) * middle_ptr

    # sanity check
    if not np.all(left_ptr == right_ptr):
        raise RuntimeError('not all errors found after ceil(log_2(l)) binsearch steps')

    return left_ptr


def get_parity_vector(key: np.array,
                      num_blocks: int,
                      start_indexes: np.array,
                      end_indexes: np.array) -> np.array:
    """
    key: permuted key - np.array (n, )
    num_blocks: number of blocks
    start_indexes: blocks start pointers
    end_indexes: blocks end pointers

    return: parity vector - np.array (num_blocks, )
    """

    parity_vector = np.zeros(num_blocks, dtype=bool)
    for i in range(num_blocks):
        parity_vector[i] = np.sum(key[start_indexes[i]: end_indexes[i]]) % 2
    return parity_vector


def recv_alices_parity_vector(client: socket, num_blocks: int) -> np.array:
    # todo научиться передавать np.array
    # msg = client.recv(num_blocks + 1).decode(ENCODING)  # socket buffer size is 8760 bytes
    #
    # if msg == DISCONNECT_MSG:
    #     log.warning(f'client disconnected: {addr}, msg: {msg}')
    #     raise RuntimeError('unexpected disconnect. msg: ' + msg)
    #
    # bob_parity_vector = np.array(list(map(int, msg)), dtype=bool)
    # log.debug(f'rcvd bobs parity vector. length: {bob_parity_vector.shape[0]}, nbytes:{bob_parity_vector.nbytes}')
    #
    # return bob_parity_vector
    pass
