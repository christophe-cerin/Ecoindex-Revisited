# lshash/storage.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# See: https://github.com/kayzhu/LSHash
#
# Ported to Python3 by Christophe Cerin (April 2023)
# Additionaly, the index method returns the bitstring corresponding to the bucket
# where we store the value.
# Usage:
"""
import sys
sys.path.append('/home/cerin/EcoIndex/LSHash/lshash')
from lshash import *
from storage import *

lsh = LSHash(6, 8)
for i in lsh.index([1,2,3,4,5,6,7,8]):
    print('Bucket id:',int(i,2))
for i in lsh.index([2,3,4,5,6,7,8,9]):
    print('Bucket id:',int(i,2))
for i in lsh.index([10,12,99,1,5,31,2,3]):
    print('Bucket id:',int(i,2))
print(lsh.query([1,2,3,4,5,6,7,7]))
"""
# and, for three executions, we get:
"""
$ python titi.py
Bucket id: 22
Bucket id: 22
Bucket id: 46
[((1, 2, 3, 4, 5, 6, 7, 8), 1), ((2, 3, 4, 5, 6, 7, 8, 9), 11)]
$ python titi.py
Bucket id: 31
Bucket id: 63
Bucket id: 31
[((1, 2, 3, 4, 5, 6, 7, 8), 1), ((10, 12, 99, 1, 5, 31, 2, 3), 10072)]
$ python titi.py
Bucket id: 52
Bucket id: 54
Bucket id: 22
[((2, 3, 4, 5, 6, 7, 8, 9), 11)]
"""
#
# The original module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

import json

try:
    import redis
except ImportError:
    redis = None

__all__ = ['storage']


def storage(storage_config, index):
    """ Given the configuration for storage and the index, return the
    configured storage instance.
    """
    if 'dict' in storage_config:
        return InMemoryStorage(storage_config['dict'])
    elif 'redis' in storage_config:
        storage_config['redis']['db'] = index
        return RedisStorage(storage_config['redis'])
    else:
        raise ValueError("Only in-memory dictionary and Redis are supported.")


class BaseStorage(object):
    def __init__(self, config):
        """ An abstract class used as an adapter for storages. """
        raise NotImplementedError

    def keys(self):
        """ Returns a list of binary hashes that are used as dict keys. """
        raise NotImplementedError

    def set_val(self, key, val):
        """ Set `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def get_val(self, key):
        """ Return `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def append_val(self, key, val):
        """ Append `val` to the list stored at `key`.

        If the key is not yet present in storage, create a list with `val` at
        `key`.
        """
        raise NotImplementedError

    def get_list(self, key):
        """ Returns a list stored in storage at `key`.

        This method should return a list of values stored at `key`. `[]` should
        be returned if the list is empty or if `key` is not present in storage.
        """
        raise NotImplementedError


class InMemoryStorage(BaseStorage):
    def __init__(self, config):
        self.name = 'dict'
        self.storage = dict()

    def keys(self):
        return self.storage.keys()

    def set_val(self, key, val):
        self.storage[key] = val

    def get_val(self, key):
        return self.storage[key]

    def append_val(self, key, val):
        self.storage.setdefault(key, []).append(val)

    def get_list(self, key):
        return self.storage.get(key, [])


class RedisStorage(BaseStorage):
    def __init__(self, config):
        if not redis:
            raise ImportError("redis-py is required to use Redis as storage.")
        self.name = 'redis'
        self.storage = redis.StrictRedis(**config)

    def keys(self, pattern="*"):
        return self.storage.keys(pattern)

    def set_val(self, key, val):
        self.storage.set(key, val)

    def get_val(self, key):
        return self.storage.get(key)

    def append_val(self, key, val):
        self.storage.rpush(key, json.dumps(val))

    def get_list(self, key):
        return self.storage.lrange(key, 0, -1)
