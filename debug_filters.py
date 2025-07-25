#!/usr/bin/env python3
"""
Debug script to check filter matrix properties
"""
import jax.numpy as jnp
import yaml
from box import Box
from ml4dynamics.dataset_utils.dataset_utils import res_int_fn

def debug_filters():
    # Load config
    with open("config/ks.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    
    N1 = config_dict["sim"]["n"] - 1  # For Dirichlet-Neumann
    r = config_dict["sim"]["rx"] 
    N2 = N1 // r
    
    print(f"N1 (fine grid): {N1}")
    print(f"N2 (coarse grid): {N2}")
    print(f"Coarsening ratio r: {r}")
    
    # Test box filter specifically
    config_dict["sim"]["filter_type"] = "box"
    
    # Manually get the restriction operator to check its properties
    from ml4dynamics.dataset_utils.dataset_utils import _create_box_filter
    
    res_op = _create_box_filter(N1, N2, r, "Dirichlet-Neumann")
    print(f"\nBox filter matrix shape: {res_op.shape}")
    print(f"Row sums (should they be 1?): {jnp.sum(res_op, axis=1)[:5]}...")  # First 5 rows
    print(f"Max row sum: {jnp.max(jnp.sum(res_op, axis=1))}")
    print(f"Min row sum: {jnp.min(jnp.sum(res_op, axis=1))}")
    
    # Try computing pseudo-inverse
    int_op = jnp.linalg.pinv(res_op)
    print(f"Interpolation matrix shape: {int_op.shape}")
    
    # Check if res_op @ int_op ≈ I
    product = res_op @ int_op
    print(f"res_op @ int_op shape: {product.shape}")
    print(f"Should be identity, diagonal elements: {jnp.diag(product)[:5]}...")
    print(f"Max off-diagonal: {jnp.max(jnp.abs(product - jnp.eye(N2)))}")
    
    # Check condition number
    print(f"Condition number of res_op: {jnp.linalg.cond(res_op)}")

if __name__ == "__main__":
    debug_filters()
