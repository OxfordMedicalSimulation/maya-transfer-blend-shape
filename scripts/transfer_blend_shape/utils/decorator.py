import itertools
from functools import wraps


__all__ = [
    "memoize",
]

TOKEN = "_memoize_token"

_counter = itertools.count()


def _identity(obj):
    """Stable per-instance identity for cache keys.

    ``str(instance)`` embeds the object's memory address, and CPython reuses
    addresses once an object is collected. Keying a cache on it means a freshly
    built object can inherit the cached geometry of a discarded one, silently
    transferring against the wrong mesh. A token stored on the instance is
    unique for its whole lifetime, so it cannot collide.

    :param obj:
    :return: Hashable identity.
    """
    try:
        token = obj.__dict__.get(TOKEN)
        if token is None:
            token = next(_counter)
            obj.__dict__[TOKEN] = token

        return token
    except AttributeError:
        # not an instance with a mutable namespace, fall back to its own repr
        return obj


def memoize(func):
    """
    The memoize decorator will cache the result of a function and store it
    in a cache dictionary using its arguments and keywords arguments as a key.
    The cache can be cleared by calling the cache_clear function on the
    decorated function.

    When decorating a method the owning instance is keyed by a unique token
    rather than its repr, see :func:`_identity`.
    """
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args and hasattr(args[0], "__dict__"):
            identity = (_identity(args[0]),) + args[1:]
        else:
            identity = args

        key = str(identity) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    def clear():
        cache.clear()

    wrapper.clear = clear
    return wrapper
