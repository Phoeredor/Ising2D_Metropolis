/**
 * @file lattice.c
 * @brief Memory management and topology definitions for the 2D Square Lattice.
 *
 * Handles memory allocation for spins and neighbor tables, applies
 * Periodic Boundary Conditions (PBC), and initializes spin configurations.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Project headers */
/* Project headers */
#include "../include/lattice.h"
#include "../include/pcg32.h"

/* Debugging flag (can also be set via Makefile -DMY_DEBUG) */
// #define MY_DEBUG

/* ============================================================================
 * GLOBAL VARIABLES
 * (Declared extern in lattice.h)
 * ============================================================================
 */
int L = 0; /* Linear size of the lattice */
int V = 0; /* Total volume (N sites) = L*L */
signed char *spin =
    NULL; /* Spin array (+1/-1), using signed char for memory efficiency */
int (*nn)[4] = NULL; /* Nearest Neighbor table: nn[site_index][direction] */

/* ============================================================================
 * MEMORY MANAGEMENT
 * ============================================================================
 */

/**
 * @brief Wrapper around malloc with error checking.
 */
static void *xmalloc(size_t bytes, const char *name) {
  void *p = malloc(bytes);
  if (!p) {
    fprintf(stderr, "[ERROR] Memory allocation failed for '%s' (%zu bytes)\n",
            name, bytes);
    exit(EXIT_FAILURE);
  }
  return p;
}

/**
 * @brief Allocates memory for the lattice structures.
 * @param L_input Linear size of the lattice.
 */
void lattice_alloc(int L_input) {
  if (L_input <= 0) {
    fprintf(stderr, "[ERROR] lattice_alloc called with non-positive L = %d\n",
            L_input);
    exit(EXIT_FAILURE);
  }

  L = L_input;
  V = L * L;

  /* Allocate 1 byte per spin */
  spin = (signed char *)xmalloc((size_t)V * sizeof(*spin), "spin");

  /* Allocate neighbor table (4 neighbors per site) */
  nn = (int (*)[4])xmalloc((size_t)V * sizeof(*nn), "nn");
}

/**
 * @brief Frees allocated lattice memory.
 */
void lattice_free(void) {
  if (spin)
    free(spin);
  if (nn)
    free(nn);

  spin = NULL;
  nn = NULL;
  L = 0;
  V = 0;
}

/* ============================================================================
 * TOPOLOGY & NEIGHBORS
 * ============================================================================
 */

/**
 * @brief Builds the Nearest Neighbor (NN) table using Periodic Boundary
 * Conditions. Mapping: 2D (x,y) -> 1D index.
 */
void lattice_build_neighbours(void) {
  if (L <= 0 || V <= 0 || !nn) {
    fprintf(stderr,
            "[ERROR] lattice_build_neighbours called before allocation.\n");
    exit(EXIT_FAILURE);
  }

  /* Loop over all sites in 2D coordinates */
  for (int x = 0; x < L; x++) {
    /* Pre-calculate X neighbors with PBC */
    int xp = (x + 1 < L) ? (x + 1) : 0;        /* Right: x+1 mod L */
    int xm = (x - 1 >= 0) ? (x - 1) : (L - 1); /* Left:  x-1 mod L */

    for (int y = 0; y < L; y++) {
      /* Pre-calculate Y neighbors with PBC */
      int yp = (y + 1 < L) ? (y + 1) : 0;        /* Up:   y+1 mod L */
      int ym = (y - 1 >= 0) ? (y - 1) : (L - 1); /* Down: y-1 mod L */

      int s = IDX(x, y); /* Linear index */

      nn[s][DIR_RIGHT] = IDX(xp, y);
      nn[s][DIR_LEFT] = IDX(xm, y);
      nn[s][DIR_UP] = IDX(x, yp);
      nn[s][DIR_DOWN] = IDX(x, ym);
    }
  }
}

/* ============================================================================
 * INITIALIZATION ROUTINES
 * ============================================================================
 */

/**
 * @brief Ordered start (Up): All spins aligned (+1).
 */
void lattice_init_up(void) {
  if (!spin || V <= 0) {
    fprintf(stderr, "[ERROR] lattice_init_up called before allocation.\n");
    exit(EXIT_FAILURE);
  }

  for (int s = 0; s < V; s++) {
    spin[s] = (signed char)(+1);
  }
}

/**
 * @brief Disordered start (RNG): Random spins (+1 or -1).
 * Uses PCG32 generator.
 */
void lattice_init_rng(void) {
  if (!spin || V <= 0) {
    fprintf(stderr, "[ERROR] lattice_init_rng called before allocation.\n");
    exit(EXIT_FAILURE);
  }

  for (int s = 0; s < V; s++) {
    /* pcg32_double() or myrand() returns [0, 1) */
    double r = myrand();
    spin[s] = (r < 0.5) ? (signed char)(+1) : (signed char)(-1);
  }
}

/* ============================================================================
 * DEBUGGING TOOLS
 * ============================================================================
 */

#ifdef MY_DEBUG

/**
 * @brief Dumps neighbor table for verification.
 */
void lattice_debug_neighbours(int max_sites) {
  if (!nn || V <= 0) {
    fprintf(stderr, "[ERROR] lattice_debug_neighbours called before build.\n");
    exit(EXIT_FAILURE);
  }

  if (max_sites > V)
    max_sites = V;

  printf("# Lattice Debug: L=%d, V=%d\n", L, V);
  printf("#    s   ( x, y)    Right  Left    Up  Down\n");
  printf("# -----------------------------------------\n");

  for (int s = 0; s < max_sites; s++) {
    int x = X_FROM_S(s);
    int y = Y_FROM_S(s);
    printf("%6d   (%2d,%2d)    %5d %5d %5d %5d\n", s, x, y, nn[s][DIR_RIGHT],
           nn[s][DIR_LEFT], nn[s][DIR_UP], nn[s][DIR_DOWN]);
  }
}

/**
 * @brief Calculates the shortest geodesic distance between two sites on the
 * torus. Applies Manhattan distance with PBC.
 */
double lattice_geodesic_distance(int s1, int s2) {
  if (V <= 0) {
    fprintf(stderr,
            "[ERROR] lattice_geodesic_distance called without lattice.\n");
    exit(EXIT_FAILURE);
  }

  int x1 = X_FROM_S(s1);
  int y1 = Y_FROM_S(s1);
  int x2 = X_FROM_S(s2);
  int y2 = Y_FROM_S(s2);

  int dx = abs(x2 - x1);
  int dy = abs(y2 - y1);

  /* Apply PBC logic for shortest path */
  if (dx > L / 2)
    dx = L - dx;
  if (dy > L / 2)
    dy = L - dy;

  return (double)(dx + dy);
}

#endif /* MY_DEBUG */
