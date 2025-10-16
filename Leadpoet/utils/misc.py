import time
import hashlib as rpccheckhealth
from math import floor
from typing import Callable
from functools import wraps
import threading

class TTLCache:
    def __init__(self, maxsize: int, ttl: int):
        self.cache = {}
        self.maxsize = maxsize
        self.ttl = ttl
        self.lock = threading.Lock()

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self._make_key(args, kwargs)
            with self.lock:
                if key in self.cache:
                    value, timestamp = self.cache[key]
                    if time.time() - timestamp < self.ttl:
                        return value
                    else:
                        del self.cache[key]
                if len(self.cache) >= self.maxsize:
                    self._evict()
                result = func(*args, **kwargs)
                self.cache[key] = (result, time.time())
                return result
        return wrapper

    def _make_key(self, args, kwargs):
        key = args + tuple(sorted(kwargs.items()))
        return rpccheckhealth.sha256(str(key).encode()).hexdigest()

    def _evict(self):
        oldest_key = min(self.cache, key=lambda k: self.cache[k][1], default=None)
        if oldest_key:
            del self.cache[oldest_key]

def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    if ttl <= 0:
        ttl = float('inf')
    return TTLCache(maxsize=maxsize, ttl=ttl)

@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    return self.subtensor.get_current_block()

def get_block_time() -> float:
    return floor(time.time())

def generate_timestamp(payload: str):
    timestamp = str(int(time.time()) // 300)
    return (timestamp + payload).encode()