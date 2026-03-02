#!/usr/bin/env python3
import sys
import numpy as np

# --- PHYSICS CONFIGURATION ---
BETA_C = 0.44068679350977151262
NU = 1.0
BETA_MIN = 0.3500
BETA_MAX = 0.5000

# Scaling variable grid for pilot runs
X_POINTS = [0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0, -5.0, 5.0, -6.0, 6.0, -7.0, 7.0, -8.0, 8.0, -9.0, 9.0, -10.0, 10.0]

def get_betas(L):
    betas = []
    for x in X_POINTS:
        beta = BETA_C + x * (L ** (-1.0 / NU))
        if BETA_MIN <= beta <= BETA_MAX:
            betas.append(beta)
            
    if BETA_MIN not in betas: betas.append(BETA_MIN)
    if BETA_MAX not in betas: betas.append(BETA_MAX)
    
    return sorted(list(set(betas)))

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    try:
        L_val = int(sys.argv[1])
        valid_betas = get_betas(L_val)
        # High-precision output (12 digits) for C program arguments
        print(" ".join([f"{b:.12f}" for b in valid_betas]))
    except ValueError:
        sys.exit(1)
