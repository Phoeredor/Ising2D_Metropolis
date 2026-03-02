/**
 * @file main_visual.c
 * @brief Snapshot-only beta sweep for visualization.
 *
 * For each beta in [beta_min, beta_max]:
 *   - thermalizes n_therm sweeps (reusing last configuration)
 *   - saves one snapshot
 * No measurement output. Much faster than main_prod.c.
 *
 * Usage: ./visual_ising <L> <beta_min> <beta_max> <n_beta> <n_therm>
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../include/ising.h"
#include "../include/lattice.h"
#include "../include/pcg32.h"
#include "../include/seed_generator.h"

/* ── helpers ─────────────────────────────────────────────────── */

static void save_snap(int L, const signed char *spin,
                      int idx, double beta)
{
    char fname[256];
    snprintf(fname, sizeof(fname),
             "snapshots/vis_L%d_%04d_beta%.4f.dat", L, idx, beta);
    FILE *fp = fopen(fname, "w");
    if (!fp) { perror("[WARN] snap open"); return; }
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++)
            fprintf(fp, "%d ", spin[i * L + j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

static long parseInt_v(const char *s) { return strtol(s, NULL, 10); }
static double parseDouble_v(const char *s) { return strtod(s, NULL); }

/* ── main ────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr,
            "Usage: %s <L> <beta_min> <beta_max> <n_beta> <n_therm>\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    int    L        = (int)parseInt_v(argv[1]);
    double beta_min = parseDouble_v(argv[2]);
    double beta_max = parseDouble_v(argv[3]);
    int    n_beta   = (int)parseInt_v(argv[4]);
    long   n_therm  = parseInt_v(argv[5]);

    /* RNG */
    seedgen_init((uint64_t)time(NULL), (uint64_t)getpid());
    myrand_init(generate_seed(), generate_seed());

    /* Lattice */
    lattice_alloc(L);
    lattice_build_neighbours();
    lattice_init_rng();            /* start disordered (high-T) */

    long E = ising_total_energy();
    long M = ising_total_magnetization();

    printf("=== Visual Beta Sweep ===\n");
    printf(" L=%d  beta: %.4f -> %.4f  n_beta=%d  n_therm=%ld\n",
           L, beta_min, beta_max, n_beta, n_therm);

    double step = (n_beta > 1)
                  ? (beta_max - beta_min) / (n_beta - 1)
                  : 0.0;

    for (int k = 0; k < n_beta; k++) {
        double beta = beta_min + k * step;

        /* Thermalize at this beta, reusing current configuration */
        for (long t = 0; t < n_therm; t++)
            ising_sweep_metropolis(beta, &E, &M);

        /* Recalculate E,M exactly after thermalisation */
        E = ising_total_energy();
        M = ising_total_magnetization();

        save_snap(L, spin, k, beta);

        if ((k + 1) % 10 == 0 || k == n_beta - 1)
            printf(" [%3d/%3d] beta=%.4f  m=%.4f\n",
                   k + 1, n_beta,
                   beta, (double)M / V);
    }

    lattice_free();
    printf("Done. %d snapshots saved.\n", n_beta);
    return EXIT_SUCCESS;
}
