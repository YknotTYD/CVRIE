## helpers.py

from typing import Callable

def get_call_str(function_like: Callable, *args, **kwargs) -> str:

    call_str = f"{function_like.__qualname__}("
    call_str += "".join([str(arg) + ", " for arg in args])
    call_str += "".join(
        [f"{key} = {value}, " for key, value in kwargs.items()]
    )

    if args or kwargs:
        call_str = call_str[:-2]

    call_str += ")"
    return call_str

def save_best(best_mean, best_std, mean, std, model) -> tuple[dict, dict]:

    if  best_mean["value"] is None or mean > best_mean["value"]:
        best_mean["value"] = mean
        best_mean["model"] = model

    if  best_std["value"] is None or std < best_std["value"]:
        best_std["value"] = std
        best_std["model"] = model

    return None
