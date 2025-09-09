#!/usr/bin/env python3
"""Simple JAX test"""

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

try:
    import jax
    print(f"✓ JAX imported successfully, version: {jax.__version__}")
    
    import jax.numpy as jnp
    print("✓ jax.numpy imported successfully")
    
    # Test computation
    x = jnp.array([1, 2, 3])
    y = jnp.sum(x)
    print(f"✓ JAX computation test: sum([1,2,3]) = {y}")
    
    print("\n🎉 JAX is working!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nThis version of JAX still has issues.")

