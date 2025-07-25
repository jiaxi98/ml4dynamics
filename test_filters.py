#!/usr/bin/env python3
"""
Test script to visualize different filter types
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import yaml
from box import Box

from ml4dynamics.dataset_utils.dataset_utils import res_int_fn

def test_filters():
    # Load config
    with open("config/ks.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    
    N1 = config_dict["sim"]["n"] - 1  # For Dirichlet-Neumann
    r = config_dict["sim"]["rx"] 
    N2 = N1 // r
    
    print(f"N1 (fine grid): {N1}")
    print(f"N2 (coarse grid): {N2}")
    print(f"Coarsening ratio r: {r}")
    
    # Test each filter type
    filter_types = ["box", "gaussian", "spectral"]
    
    fig, axes = plt.subplots(len(filter_types), 1, figsize=(12, 8))
    
    for idx, filter_type in enumerate(filter_types):
        print(f"\nTesting {filter_type} filter...")
        
        # Update config for this filter type
        config_dict["sim"]["filter_type"] = filter_type
        
        # Get filter operators
        res_fn, int_fn = res_int_fn(config_dict)
        
        # Test with a delta function in the middle
        test_signal = jnp.zeros(N1)
        test_signal = test_signal.at[N1//2].set(1.0)
        
        # Apply restriction (filtering)
        filtered = res_fn(test_signal)
        print(f"Input max: {jnp.max(test_signal):.3f}, Output max: {jnp.max(filtered):.3f}")
        
        # Plot
        ax = axes[idx]
        x_fine = jnp.linspace(0, 128, N1)  # Using L=128 from config
        x_coarse = jnp.linspace(0, 128, N2)
        
        ax.plot(x_fine, test_signal, 'b-', alpha=0.7, label='Fine grid input')
        ax.plot(x_coarse, filtered[:, 0], 'ro-', label=f'{filter_type.capitalize()} filtered')
        ax.set_title(f'{filter_type.capitalize()} Filter Response')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('filter_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nFilter comparison saved to filter_comparison.png")

if __name__ == "__main__":
    test_filters()
