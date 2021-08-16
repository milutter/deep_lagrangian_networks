import jax
import jax.numpy as jnp
import haiku as hk

def parition_params(module_name, name, value, key):
    return module_name.split("/")[0] == key

def get_params(params, key):
    return hk.data_structures.partition(jax.partial(parition_params, key=key), params)

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
}
