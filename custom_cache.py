import time

class MotionCache:
    def __init__(self):
        self.store = {}  # key -> (value, expiry_time)

    def set(self, key, value, ttl_seconds=10):
        """Set a key with custom TTL"""
        expiry = time.time() + ttl_seconds
        self.store[key] = (value, expiry)

    def get(self, key, default=None):
        """Get a key, return default if expired or missing"""
        if key in self.store:
            value, expiry = self.store[key]
            if time.time() < expiry:
                return value
            else:
                # expired, remove it
                del self.store[key]
        return default
    