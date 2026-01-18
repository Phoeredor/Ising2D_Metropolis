/* * ISING METROPOLIS PHYSICS TEST
 * Verifies energy/magnetization tracking and update mechanics.
 *
 * Compile from root:
 * gcc -Wall -O2 -Iinclude tests/test_physics.c src/lattice.c src/ising.c src/pcg32.c src/seed_generator.c -lm -o tests/physics_check
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lattice.h"
#include "ising.h"
#include "pcg32.h"    

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stdout, "Usage: %s <L> <beta> <n_sweeps>\n", argv[0]);
        return EXIT_FAILURE;
    }
        
    int L_input = atoi(argv[1]);
    double beta = atof(argv[2]);
    long n_sweeps = atol(argv[3]);

    if (L_input <= 0 || n_sweeps <= 0) {
        fprintf(stderr, "Error: L and n_sweeps must be positive.\n");
        return EXIT_FAILURE;
    }

    // 1. Init RNG with fixed seed for reproducibility during test
    myrand_init(12345u, 67890u);

    // 2. Setup Lattice
    lattice_alloc(L_input);
    lattice_build_neighbours();
    lattice_init_hot(); // Random start

    // 3. Initial Measurements (Full calculation)
    long E_tot = ising_total_energy();
    long M_tot = ising_total_magnetization();

    printf("\n=== ISING PHYSICS TEST ===\n");
    printf("L=%d, beta=%.4f, sweeps=%ld\n", L_input, beta, n_sweeps);
    printf("Initial: E=%ld, M=%ld\n", E_tot, M_tot);

    // 4. Run Sweeps
    for (long sweep = 1; sweep <= n_sweeps; sweep++) {
        // Pass pointers to E and M so they are updated incrementally
        double acc_rate = ising_sweep_metropolis(beta, &E_tot, &M_tot);

        // Periodically verify incremental updates against full recalculation
        if (sweep == 1 || sweep == n_sweeps || sweep % 1000 == 0) {
            long E_check = ising_total_energy();
            long M_check = ising_total_magnetization();
            
            double e_dens = (double)E_tot / V;
            double m_dens = (double)M_tot / V;

            printf("Sweep %6ld: Acc=%.3f | e=%.5f m=%.5f", sweep, acc_rate, e_dens, m_dens);

            if (E_tot != E_check || M_tot != M_check) {
                printf(" [DRIFT ERROR!]\n");
                fprintf(stderr, "FATAL: Drift detected. Tracker(E=%ld, M=%ld) != Full(E=%ld, M=%ld)\n",
                        E_tot, M_tot, E_check, M_check);
                return EXIT_FAILURE;
            } else {
                printf(" [OK]\n");
            }
        }
    }

    lattice_free();
    printf("\n[SUCCESS] Physics loop completed without drift.\n");
    return EXIT_SUCCESS;
}
