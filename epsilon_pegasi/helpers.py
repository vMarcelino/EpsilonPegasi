from dataclasses import dataclass
from functools import wraps
from typing import List, Union, Dict


def _try_eval(code: str):
    try:
        return eval(code)
    except:
        return lambda x: x


def _enforced_dataclass(cls,
                        exceptions: List[Union[type, str]],
                        replaces: Dict[str, callable],
                        blacklist: bool,
                        debug: bool = False):
    if exceptions is None:
        exceptions = []
    if replaces is None:
        replaces = {}
    dataclass(cls)
    init = cls.__init__
    annotations = init.__annotations__
    exceptions_str = [r if type(r) is str else r.__name__ for r in exceptions]
    replaces_str = {t if type(t) is str else t.__name__: r for t, r in replaces.items()}

    @wraps(init)
    def __init__(*args, **kwargs):
        # converts args to kwargs
        kwargs.update(zip(init.__code__.co_varnames, args))

        def enforce_type(kwarg):
            value = kwargs[kwarg]
            enforce_func = annotations[kwarg]
            if type(enforce_func) is str:
                if enforce_func in replaces_str:
                    enforce_func = replaces_str[enforce_func]
                    if debug: print('using custom enforce function:', enforce_func)
                else:
                    enforce_func = _try_eval(enforce_func)
            else:
                if enforce_func in replaces:
                    enforce_func = replaces[enforce_func]
                    if debug: print('using custom enforce function:', enforce_func)
                elif enforce_func.__name__ in replaces_str:
                    enforce_func = replaces_str[enforce_func.__name__]
                    if debug: print('using custom enforce function:', enforce_func)

            return enforce_func(value)

        def type_check(kwarg):
            mapper = annotations[kwarg]
            value = kwargs[kwarg]
            if type(mapper) is str:
                #check by type name
                if type(value).__name__ != mapper:
                    # enforce type
                    kwargs[kwarg] = enforce_type(kwarg)
                    if debug: print('enforced to', repr(kwargs[kwarg]))

            else:
                # check by type
                if type(value) is not mapper:
                    #enforce type
                    kwargs[kwarg] = enforce_type(kwarg)
                    if debug: print('enforced to', repr(kwargs[kwarg]))

        # enforces types
        for kwarg, value in kwargs.items():
            if kwarg in annotations:
                if debug: print('found', kwarg, 'as', repr(value), 'of type', type(value))
                mapper = annotations[kwarg]
                is_exc = False
                if type(mapper) is str:
                    is_exc = mapper in exceptions_str
                else:
                    is_exc = mapper in exceptions or mapper.__name__ in exceptions_str

                if is_exc:
                    if blacklist:
                        # do not check
                        if debug: print('not enforcing')
                        pass
                    else:
                        # check
                        type_check(kwarg)
                        pass
                else:
                    if blacklist:
                        # check
                        type_check(kwarg)
                        pass
                    else:
                        # do not check
                        if debug: print('not enforcing')
                        pass

        return init(**kwargs)

    cls.__init__ = __init__
    return cls


def enforced_dataclass(cls=None,
                       *,
                       exceptions: List[Union[type, str]] = None,
                       replaces: Dict[str, callable] = None,
                       blacklist: bool = True,
                       debug: bool = False):
    if cls:
        return _enforced_dataclass(cls, [], {}, True, debug)
    else:
        return lambda _cls: _enforced_dataclass(_cls, exceptions, replaces, blacklist, debug)
