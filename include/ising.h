/**
 * @file ising.h
 * @brief Physics definitions for the 2D Ising Model.
 *
 * Defines the Hamiltonian, observables, and the Monte Carlo update step.
 * Assumes a square lattice defined in lattice.h.
 */

#ifndef ISING_H
#define ISING_H

#include "lattice.h"  /* Requires global V variable */

/* ============================================================================
 * OBSERVABLES (Full Calculation)
 * ============================================================================ */

/**
 * @brief Calculates the total energy of the current configuration.
 * * Hamiltonian: H = -J * sum_<i,j> (s_i * s_j)
 * with J=1 and h=0 (no external field).
 * Complexity: O(V)
 * * @return Total Energy (long integer).
 */
long ising_total_energy(void);

/**
 * @brief Calculates the total magnetization of the current configuration.
 * * M = sum_i s_i
 * Complexity: O(V)
 * * @return Total Magnetization (long integer).
 */
long ising_total_magnetization(void);


/* ============================================================================
 * INLINE HELPERS
 * Warning: These perform O(V) calculations. Do not use in critical loops.
 * ============================================================================ */

/**
 * @brief Returns current Energy per spin.
 */
static inline double ising_energy_density(void)
{
    return (double)ising_total_energy() / (double)V;
}

/**
 * @brief Returns current Magnetization per spin.
 */
static inline double ising_magnetization_density(void)
{
    return (double)ising_total_magnetization() / (double)V;
}


/* ============================================================================
 * DYNAMICS (Metropolis Algorithm)
 * ============================================================================ */

/**
 * @brief Performs one full Metropolis sweep over the lattice.
 * * Attempts V spin flips sequentially.
 * - If dE <= 0: Accept.
 * - If dE > 0: Accept with probability exp(-beta * dE).
 * * Updates the passed Energy and Magnetization pointers incrementally
 * to avoid expensive O(V) recalculations.
 * * @param beta Inverse temperature (1/T).
 * @param E Pointer to current Energy (updated in place).
 * @param M Pointer to current Magnetization (updated in place).
 * @return Acceptance rate of the sweep (0.0 to 1.0).
 */
double ising_sweep_metropolis(double beta, long *E, long *M);

#endif /* ISING_H */
