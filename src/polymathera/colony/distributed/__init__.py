


_polymathera = None

def get_polymathera():
    global _polymathera
    if _polymathera is None:
        from .system import polymathera # break import cycles
        _polymathera = polymathera
    return _polymathera


async def get_initialized_polymathera():
    _polymathera = get_polymathera()
    await _polymathera.initialize()
    return _polymathera

