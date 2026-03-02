/*
 * GEOMETRY LATTICE TEST PROGRAM
 * Verifies PBC (Periodic Boundary Conditions) and neighbor connectivity.
 *
 * Compile from root:
 * gcc -Wall -O2 -Iinclude tests/test_lattice.c src/lattice.c src/pcg32.c src/seed_generator.c -DMY_DEBUG -o tests/lattice_check
 */

#include <stdio.h>
#include <stdlib.h>
#include "lattice.h"

/* Helper: Get opposite direction */
static int opposite_dir(int dir) {
    switch (dir) {
        case DIR_RIGHT: return DIR_LEFT;
        case DIR_LEFT:  return DIR_RIGHT;
        case DIR_UP:    return DIR_DOWN;
        case DIR_DOWN:  return DIR_UP;
        default: return -1;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <L>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    int L_input = atoi(argv[1]);
    if (L_input <= 0) {
        fprintf(stderr, "Error: L must be positive.\n");
        return EXIT_FAILURE;
    }
    
    printf("\n=== LATTICE GEOMETRY TEST (L=%d) ===\n", L_input);
    
    // 1. Allocation
    lattice_alloc(L_input);
    
    // 2. Build Neighbors
    lattice_build_neighbours(); // British spelling as per lattice.h
    
    // 3. Init (Cold)
    lattice_init_cold();
    
    // 4. Debug Print (First few sites)
    // Requires compilation with -DMY_DEBUG
    #ifdef MY_DEBUG
    int max_sites = (V < 16) ? V : 16;
    printf("--- First %d sites connectivity ---\n", max_sites);
    lattice_debug_neighbours(max_sites); 
    #endif
    
    // 5. Topology Check: Go there and come back
    int errors = 0;
    for (int s = 0; s < V; s++) {
        for (int dir = 0; dir < 4; dir++) {
            int neighbor = nn[s][dir];
            int back_dir = opposite_dir(dir);
            
            // Check bounds
            if (neighbor < 0 || neighbor >= V) {
                fprintf(stderr, "[FAIL] Site %d dir %d points to invalid %d\n", s, dir, neighbor);
                errors++;
                continue;
            }

            // Check reversibility
            int origin = nn[neighbor][back_dir];
            if (origin != s) {
                fprintf(stderr, "[FAIL] PBC Broken: %d -> %d (dir %d), but %d -> %d (back)\n",
                        s, neighbor, dir, neighbor, origin);
                errors++;
            }
        }
    }
    
    // 6. Geodesic Check: Distance to self must be 0
    #ifdef MY_DEBUG
    for (int s = 0; s < V; s++) {
        double d = lattice_geodesic_distance(s, s);
        if (d != 0.0) {
            fprintf(stderr, "[FAIL] Non-zero self-distance: site %d dist %g\n", s, d);
            errors++;
        }
    }
    #endif

    lattice_free();
    
    if (errors == 0) {
        printf("\n[SUCCESS] All lattice topology tests passed.\n");
        return EXIT_SUCCESS;
    } else {
        printf("\n[FAIL] Found %d topology errors.\n", errors);
        return EXIT_FAILURE;
    }
}
