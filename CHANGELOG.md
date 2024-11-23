# Change Log
## 2024-11-21
Changes by Brandon Johns.
Bugfixes and removed use of latex.

### Added
- `requirements-cpu.txt` file because the JAX devs are in a hurry to deprecate code

### Changed
- Removed use of latex from plots
- Added some more instructions to the readme

### Fixed
- Internal Bug
    - In "example_DeLaN.py", "load_dataset()" had a mismatching number of output arguments
- Matplotlib deprecations 1
    - Replace ```fig.canvas.set_window_title```
    - with    ```fig.canvas.manager.set_window_title```
- Matplotlib deprecations 2
    - Replace ```cm.get_cmap```
    - with    ```matplotlib.colormaps.get_cmap```
    - and add ```import matplotlib``` to the top of each file that uses it
- JAX deprecations 1
    - See: https://github.com/jax-ml/jax/issues/11280
    - Replace ```jax.partial```
    - with ```functools.partial```
    - and add ```import functools``` to the top of each file that uses it
- JAX deprecations 2
    - See: https://github.com/jax-ml/jax/issues/11706
    - Remove ```mat_idx = jax.ops.index[..., mat_idx[0], mat_idx[1]]```
    - and
    - Replace ```triangular_mat = jax.ops.index_update(triangular_mat, mat_idx, vec_lower_triangular[:])```
    - with    ```triangular_mat = triangular_mat.at[mat_idx].set(vec_lower_triangular[:])```
- JAX deprecations 3
    - Replace ```jax.tree_map```
    - with    ```jax.tree.map```



