/**
 * @file ising.c
 * @brief Core implementation of the 2D Ising Model physics.
 *
 * This file contains the Hamiltonian calculation, magnetization tracking,
 * and the Metropolis-Hastings update step. It includes optional debugging
 * and statistical validation features.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Project headers */
#include "lattice.h"
#include "ising.h"
#include "pcg32.h"
#include "seed_generator.h"

/* ============================================================================
 * DEBUGGING & VALIDATION FLAGS
 * Uncomment to enable heavy validation checks.
 * WARNING: These will significantly slow down simulation!
 * ============================================================================ */
//#define MY_DEBUG              // Enable per-step verbose logging
//#define DEBUG_SITE_VISITS     // Enable statistical analysis of site selection (Poisson check)


/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * @brief Sums the spins of the 4 nearest neighbors.
 * @param s Index of the site.
 * @return Sum of neighbor spins (range: -4 to +4).
 */
static inline int nn_sum(int s)
{
    int sum = 0;
    sum += spin[nn[s][DIR_RIGHT]];
    sum += spin[nn[s][DIR_LEFT]];
    sum += spin[nn[s][DIR_UP]];
    sum += spin[nn[s][DIR_DOWN]];
    return sum;
}

/* * Look-up Boltzmann table optimization.
 * For 2D Ising (J=1, h=0), ΔE ∈ {-8, -4, 0, +4, +8}.
 * We only need probabilities for ΔE > 0, i.e., ΔE = 4 and ΔE = 8.
 */
static double bolt_table[3];   // idx 0: ΔE=0 (unused), 1: ΔE=4, 2: ΔE=8
static double current_beta = -1.0;

/* Macro to map ΔE (4 or 8) to index (1 or 2) */
#define BOLT_IDX_POS(dE)  ((dE) / 4)

static inline void update_boltzmann_table(double beta)
{
    if (beta == current_beta) {
        return;
    }

    bolt_table[0] = 1.0;                  /* ΔE = 0 (not used) */
    bolt_table[1] = exp(-4.0 * beta);     /* ΔE = 4 */
    bolt_table[2] = exp(-8.0 * beta);     /* ΔE = 8 */

    current_beta = beta;
}


/* ============================================================================
 * PHYSICS CALCULATIONS
 * ============================================================================ */

/**
 * @brief Calculates the total energy of the system.
 * H = - sum_<i,j> s_i s_j
 * We iterate over all sites and sum bonds to the RIGHT and UP to avoid double counting.
 */
long ising_total_energy(void)
{
    if (V <= 0 || !spin || !nn) {
        fprintf(stderr, "[ERROR] ising_total_energy called without valid lattice.\n");
        exit(EXIT_FAILURE);
    }

    long E = 0;

    for (int s = 0; s < V; s++) {
        int right = nn[s][DIR_RIGHT];
        int up    = nn[s][DIR_UP];

        int si = (int)spin[s];
        
        // Interaction terms
        E += -si * (int)spin[right];
        E += -si * (int)spin[up];
    }

    return E;
}

/**
 * @brief Calculates the total magnetization of the system.
 * M = sum_i s_i
 */
long ising_total_magnetization(void)
{
    if (V <= 0 || !spin) {
        fprintf(stderr, "[ERROR] ising_total_magnetization called without valid lattice.\n");
        exit(EXIT_FAILURE);
    }

    long M = 0;
    for (int s = 0; s < V; s++) {
        M += (int)spin[s];
    }

    return M;
}


/* ============================================================================
 * METROPOLIS ALGORITHM
 * ============================================================================ */

/**
 * @brief Performs one full Metropolis sweep over the lattice.
 * * Algorithm:
 * 1. For V attempts:
 * 2. Pick a random site.
 * 3. Calculate energy change ΔE.
 * 4. Accept flip if ΔE <= 0 or with probability exp(-βΔE).
 * * @param beta Inverse temperature.
 * @param E Pointer to current Energy (updated in place).
 * @param M Pointer to current Magnetization (updated in place).
 * @return Acceptance rate (0.0 to 1.0).
 */
double ising_sweep_metropolis(double beta, long *E, long *M)
{
    if (V <= 0 || !spin || !nn) {
        fprintf(stderr, "[ERROR] ising_sweep_metropolis: invalid lattice.\n");
        exit(EXIT_FAILURE);
    }
    if (!E || !M) {
        fprintf(stderr, "[ERROR] ising_sweep_metropolis: NULL pointers for E/M.\n");
        exit(EXIT_FAILURE);
    }

    /* --------------------------------------------------------------------
     * DEBUG: Site Visit Tracking Initialization
     * -------------------------------------------------------------------- */
    #ifdef DEBUG_SITE_VISITS
    static int *visit_count = NULL;
    static int prev_V = 0;
    static long sweep_counter = 0;

    if (visit_count == NULL || prev_V != V) {
        free(visit_count);
        visit_count = (int *)calloc(V, sizeof(int));
        if (!visit_count) {
            fprintf(stderr, "[ERROR] Failed to allocate visit_count debug array.\n");
            exit(EXIT_FAILURE);
        }
        prev_V = V;
        sweep_counter = 0;
    }

    // Reset counter at start of each sweep
    memset(visit_count, 0, V * sizeof(int));
    sweep_counter++;
    #endif
    /* -------------------------------------------------------------------- */

    // Pre-compute exponentials if beta changed
    update_boltzmann_table(beta);

    int accepted = 0;

    for (int step = 0; step < V; step++) {
        // Select random site: Uses high-quality PCG32 generator
        int s = (int)(pcg32_random() % (uint32_t)V);

        #ifdef DEBUG_SITE_VISITS
        visit_count[s]++;
        #endif

        int s_old  = (int)spin[s];
        int sum_nn = nn_sum(s);

        // Calculate changes
        int dE_int = 2 * s_old * sum_nn;  // ΔE ∈ {-8, -4, 0, 4, 8}
        int dM_int = -2 * s_old;          // ΔM ∈ {-2, 2}

        #ifdef MY_DEBUG
        if (dE_int < -8 || dE_int > 8 || (dE_int % 4) != 0) {
            fprintf(stderr, "[ERROR] Unexpected dE = %d\n", dE_int);
            exit(EXIT_FAILURE);
        }
        #endif

        // Metropolis Criterion
        if (dE_int <= 0) {
            // Energy decreases or stays same -> Accept always
            spin[s] = (signed char)(-s_old);
            *E += (long)dE_int;
            *M += (long)dM_int;
            accepted++;
        } 
        else {
            // Energy increases -> Accept with probability P = exp(-βΔE)
            int idx  = BOLT_IDX_POS(dE_int); 
            double w = bolt_table[idx];
            double r = myrand(); // Uniform [0, 1)

            #ifdef MY_DEBUG
            static int debug_counter = 0;
            if (debug_counter < 20 && idx > 0) {
                printf("[DEBUG] dE_idx=%d, w=%.4f, r=%.4f, Accepted? %d\n", 
                       idx, w, r, (r < w));
                debug_counter++;
            }
            #endif

            if (r < w) {
                spin[s] = (signed char)(-s_old);
                *E += (long)dE_int;
                *M += (long)dM_int;
                accepted++;
            }
        }
    }

    /* --------------------------------------------------------------------
     * DEBUG: Statistical Analysis of Site Visits (Poisson Check)
     * -------------------------------------------------------------------- */
    #ifdef DEBUG_SITE_VISITS
    // Perform analysis every 100 sweeps
    if (sweep_counter % 100 == 0) {
        int max_visits = 0;
        for (int s = 0; s < V; s++) {
            if (visit_count[s] > max_visits) max_visits = visit_count[s];
        }

        int *histogram = (int *)calloc(max_visits + 1, sizeof(int));
        for (int s = 0; s < V; s++) {
            histogram[visit_count[s]]++;
        }

        printf("\n=== SITE VISIT DISTRIBUTION (Sweep %ld) ===\n", sweep_counter);
        printf("k_visits  n_sites   fraction  expected(Poisson)\n");
        printf("-----------------------------------------------\n");

        for (int k = 0; k <= max_visits; k++) {
            if (histogram[k] > 0) {
                double frac_obs = (double)histogram[k] / (double)V;
                
                // Poisson(λ=1) calculation: P(X=k) = e^(-1) / k!
                double poisson = exp(-1.0);
                for (int i = 1; i <= k; i++) poisson /= i;

                printf("%8d  %7d  %8.4f  %8.4f", k, histogram[k], frac_obs, poisson);

                // Statistical deviation check (3-sigma)
                double expected = V * poisson;
                double sigma = sqrt(expected * (1.0 - poisson));
                double z = fabs(histogram[k] - expected) / sigma;
                
                if (z > 3.0) {
                    printf("  *** DEVIATION %.1fσ ***", z);
                }
                printf("\n");
            }
        }

        // Summary
        int not_visited = histogram[0];
        int visited_once = histogram[1];
        int multi_visit = V - not_visited - visited_once;

        printf("\nSummary:\n");
        printf("  Not visited:      %d (%.1f%%, Expect %.1f%%)\n", 
               not_visited, 100.0 * not_visited / V, 100.0 * exp(-1.0));
        printf("  Visited once:     %d (%.1f%%, Expect %.1f%%)\n", 
               visited_once, 100.0 * visited_once / V, 100.0 * exp(-1.0));
        printf("  Visited multiple: %d (%.1f%%, Expect %.1f%%)\n", 
               multi_visit, 100.0 * multi_visit / V, 100.0 * (1.0 - 2.0 * exp(-1.0)));
        printf("===============================================\n\n");

        free(histogram);
    }
    #endif
    /* -------------------------------------------------------------------- */

    return (double)accepted / (double)V;
}
