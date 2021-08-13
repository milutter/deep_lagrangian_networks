import jax
import haiku as hk

def parition_params(module_name, name, value, key):
    return module_name.split("/")[0] == key

def get_params(params, key):
    return hk.data_structures.partition(jax.partial(parition_params, key=key), params)
