from dataclasses import dataclass
from functools import wraps
from typing import List, Union


def _try_eval(code: str):
    try:
        return eval(code)
    except:
        return None


def _enforced_dataclass(cls, exceptions: List[Union[type, str]], blacklist: bool, debug: bool = False):
    dataclass(cls)
    init = cls.__init__
    _annotations = init.__annotations__
    annotations = {}
    for k, v in _annotations.items():
        if callable(v):
            annotations[k] = v
        else:
            ev = _try_eval(v)
            if callable(ev):
                annotations[k] = ev

    for i, e in enumerate(exceptions):
        if type(e) is str:
            exceptions[i] = eval(e)

    @wraps(init)
    def __init__(*args, **kwargs):
        # converts args to kwargs
        kwargs.update(zip(init.__code__.co_varnames, args))

        # enforces types
        for kwarg in kwargs:
            if kwarg in annotations:
                if debug: print('found', kwarg, 'as', kwargs[kwarg])
                if (annotations[kwarg] in exceptions) ^ blacklist:
                    kwargs[kwarg] = annotations[kwarg](kwargs[kwarg])
                    if debug: print('enforcing to ', annotations[kwarg])

        return init(**kwargs)

    cls.__init__ = __init__
    return cls


def enforced_dataclass(cls=None,
                       *,
                       exceptions: List[Union[type, str]] = None,
                       blacklist: bool = False,
                       debug: bool = False):
    if cls:
        return _enforced_dataclass(cls, [], True, debug)
    else:
        return lambda _cls: _enforced_dataclass(_cls, exceptions, blacklist, debug)
