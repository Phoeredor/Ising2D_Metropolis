/**
 * @file lattice.h
 * @brief Geometry and memory definitions for the 2D Square Lattice.
 *
 * Defines the global state variables (L, V, spin array) and the topology
 * macros for mapping 2D coordinates to 1D linear indices.
 * Assumes Periodic Boundary Conditions (PBC).
 */

#ifndef LATTICE_H
#define LATTICE_H

#include <stddef.h> /* For size_t */

/* ============================================================================
 * GLOBAL SIMULATION STATE
 * Defined in lattice.c
 * ============================================================================
 */

extern int L; /**< Linear size of the lattice */
extern int V; /**< Total volume (N sites) = L * L */

/**
 * @brief Spin configuration array.
 * Values: +1 or -1. Stored as signed char for memory efficiency.
 */
extern signed char *spin;

/**
 * @brief Nearest Neighbor table.
 * nn[s][dir] gives the index of the neighbor of site 's' in direction 'dir'.
 */
extern int (*nn)[4];

/* ============================================================================
 * TOPOLOGY DEFINITIONS
 * ============================================================================
 */

/* Direction Indices (Fixed Convention) */
#define DIR_RIGHT 0 /* (x+1, y) */
#define DIR_LEFT 1  /* (x-1, y) */
#define DIR_UP 2    /* (x, y+1) */
#define DIR_DOWN 3  /* (x, y-1) */

/* * Coordinate Mapping Macros
 * s = x*L + y  (where 0 <= x,y < L)
 */

/** @brief Maps 2D coordinates (x,y) to 1D linear index. */
#define IDX(x, y) ((int)((x) * L + (y)))

/** @brief Extracts X coordinate from linear index s. */
#define X_FROM_S(s) ((int)((s) / L))

/** @brief Extracts Y coordinate from linear index s. */
#define Y_FROM_S(s) ((int)((s) % L))

/* ============================================================================
 * MEMORY & INITIALIZATION
 * ============================================================================
 */

/**
 * @brief Allocates memory for lattice structures (spins and neighbors).
 * Sets global L and V.
 * @param L_input Linear size of the lattice.
 */
void lattice_alloc(int L_input);

/**
 * @brief Builds the nearest-neighbor table (nn) applying Periodic Boundary
 * Conditions. Must be called after lattice_alloc().
 */
void lattice_build_neighbours(void);

/**
 * @brief Initializes spins to an "Up" start (Ordered).
 * All spins set to +1.
 */
void lattice_init_up(void);

/**
 * @brief Initializes spins to an "RNG" start (Disordered).
 * Spins are randomized (+1/-1) with 50% probability.
 * Requires initialized RNG.
 */
void lattice_init_rng(void);

/**
 * @brief Frees all lattice-related memory.
 */
void lattice_free(void);

/* ============================================================================
 * DEBUGGING & DIAGNOSTICS
 * ============================================================================
 */

/**
 * @brief Dumps geometry and neighbor info for the first 'max_sites'.
 * Useful for verifying PBC implementation.
 */
void lattice_debug_neighbours(int max_sites);

/**
 * @brief Calculates the shortest geodesic distance between two sites on the
 * torus. Accounts for Periodic Boundary Conditions.
 * @return Manhattan distance (dx + dy) corrected for PBC.
 */
double lattice_geodesic_distance(int s1, int s2);

#endif /* LATTICE_H */
