import numpy as np
import socket
import math
import numpy_indexed as npi
import logging as log

# logging config
stream_handler = log.StreamHandler()
stream_handler.setLevel(log.INFO)
log.basicConfig(level=log.DEBUG,
                format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                handlers=[log.FileHandler('test_connection_alice_log.log', mode='w'),
                          stream_handler])

# cascade config
SEED = 1337
MAX_ITER = 4
MAGIC_CONSTANT = 0.73

# network config
ENCODING = 'utf-8'
CONNECTION_TIMEOUT = 20
DISCONNECT_MSG = '!DISCONNECT'


