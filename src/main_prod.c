/**
 * @file main_prod.c
 * @brief Production Simulation for 2D Ising Model.
 *
 * This program handles high-statistics data acquisition using the Metropolis
 * algorithm. Features:
 * - Binary/ASCII output switching.
 * - Robust input parsing.
 * - Drift detection (integrity check).
 * - Snapshot capabilities for visualization.
 * - Thermalization logging.
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* Project headers */
#include "../include/ising.h"
#include "../include/lattice.h"
#include "../include/pcg32.h"
#include "../include/seed_generator.h"

/* ============================================================================
 * CONFIGURATION MACROS
 * ============================================================================
 */

/* Output Format Control (Default: Binary) */
/* Can be overridden by Makefile: -UUSE_BINARY_OUTPUT */
#ifndef USE_BINARY_OUTPUT
#define USE_BINARY_OUTPUT
#endif

/* Feature Flags */
// #define MY_DEBUG             // Enable heavy internal integrity checks (Drift
// detection)
#define ENABLE_SNAPSHOTS // Enable lattice configuration saving
// #define THERMALIZATION_TEST  // Enable verbose logging during thermalization
// (skips production)

/* File Configuration */
#ifdef USE_BINARY_OUTPUT
#define FILE_EXT "bin"
#define FILE_MODE "wb"
#else
#define FILE_EXT "dat"
#define FILE_MODE "w"
#endif

/* ============================================================================
 * STRUCTS & PROTOTYPES
 * ============================================================================
 */

typedef struct {
  int L;
  double beta;
  long n_therm;
  long n_sweeps;
  long meas_stride;
  char init_type[10];
} sim_params_t;

/* Binary Header (Packed for portability) */
#pragma pack(push, 1)
typedef struct {
  int32_t L;
  int32_t _pad;
  double beta;
  int64_t n_therm;
  int64_t n_sweeps;
  int64_t meas_stride;
  uint32_t seed1;
  uint32_t seed2;
} header_t;

/* Measurement Record */
typedef struct {
  int64_t sweep;
  double e;
  double m;
  double e2;
  double m2;
  double m4;
} measurement_t;
#pragma pack(pop)

/* Parsing Helper Prototypes */
int parseInt(const char *arg, const char *name);
long parseLong(const char *arg, const char *name);
double parseDouble(const char *arg, const char *name);
static sim_params_t parse_args(int argc, char **argv);

/* Snapshot Helper */
void save_snapshot(int L, const signed char *spin, int snapshot_idx,
                   double beta);

/* ============================================================================
 * MAIN FUNCTION
 * ============================================================================
 */

int main(int argc, char **argv) {
  /* 1. Input Parsing */
  sim_params_t P = parse_args(argc, argv);

  /* 2. RNG Initialization (Time + PID for Entropy) */
  uint64_t time_seed = (uint64_t)time(NULL);
  uint64_t pid_seed = (uint64_t)getpid();
  seedgen_init(time_seed, pid_seed);

  unsigned int seed1 = generate_seed();
  unsigned int seed2 = generate_seed();
  myrand_init(seed1, seed2);

  /* 3. System Setup */
  lattice_alloc(P.L);
  lattice_build_neighbours();

  if (strcmp(P.init_type, "rng") == 0) {
    lattice_init_rng();
  } else {
    lattice_init_up();
  }

  long E = ising_total_energy();
  long M = ising_total_magnetization();

  /* 4. Console Banner */
  printf("========================================\n");
  printf("|   ISING 2D - PRODUCTION SIMULATION   |\n");
  printf("========================================\n");
  printf(" L             = %d  (V = %d)\n", P.L, V);
  printf(" beta          = %.6f\n", P.beta);
  printf(" n_therm       = %ld\n", P.n_therm);
  printf(" n_sweeps      = %ld\n", P.n_sweeps);
  printf(" meas_stride   = %ld\n", P.meas_stride);
  printf(" init_type     = %s\n", P.init_type);
#ifdef USE_BINARY_OUTPUT
  printf(" output        = BINARY (.bin)\n");
#else
  printf(" output        = ASCII (.dat)\n");
#endif
  printf("----------------------------------------\n");

  /* 5. Output File Setup */
  char fname[256];
  /* File is saved in current working directory (managed by wrapper script) */
  snprintf(fname, sizeof(fname), "%s_obs_L%d_beta%.6f.%s", P.init_type, P.L,
           P.beta, FILE_EXT);

  FILE *fp = fopen(fname, FILE_MODE);
  if (!fp) {
    perror("Error opening output file");
    exit(EXIT_FAILURE);
  }

/* Write Header */
#ifdef USE_BINARY_OUTPUT
  header_t header = {P.L,           0,     P.beta, P.n_therm, P.n_sweeps,
                     P.meas_stride, seed1, seed2};
  fwrite(&header, sizeof(header_t), 1, fp);
#else
  fprintf(fp,
          "# L=%d beta=%.6f n_therm=%ld n_sweeps=%ld stride=%ld s1=%u s2=%u\n",
          P.L, P.beta, P.n_therm, P.n_sweeps, P.meas_stride, seed1, seed2);
  fprintf(fp, "# sweep e m e2 m2 m4\n");
#endif

  /* 6. Thermalization Phase */
  double acc_sum_therm = 0.0;
  long acc_count_therm = 0;

/* Optional: Thermalization logging file */
#ifdef THERMALIZATION_TEST
  char fname_test[256];
  snprintf(fname_test, sizeof(fname_test), "therm_log_L%d_beta%.3f.dat", P.L,
           P.beta);
  FILE *fp_test = fopen(fname_test, "w");
#endif

  for (long t = 0; t < P.n_therm; t++) {
    double acc = ising_sweep_metropolis(P.beta, &E, &M);
    acc_sum_therm += acc;
    acc_count_therm++;

    /* Logging */
    if ((t + 1) % 1000 == 0) {
      double acc_mean = acc_sum_therm / (double)acc_count_therm;
      printf("Therm sweep %ld | Acc = %.3f\n", t + 1, acc_mean);
    }

#ifdef THERMALIZATION_TEST
    if ((t + 1) % 100 == 0 && fp_test) {
      double e = (double)E / V;
      double m = (double)M / V;
      fprintf(fp_test, "%ld %.6f %.6f %.6f %.3f\n", t + 1, e, fabs(m), e * e,
              acc);
    }
#endif
  }

#ifdef THERMALIZATION_TEST
  if (fp_test)
    fclose(fp_test);
  printf(">> Thermalization Test Finished. Exiting.\n");
  fclose(fp);
  lattice_free();
  return EXIT_SUCCESS;
#endif

  printf(">> Thermalization Done. Avg Acc = %.4f\n",
         acc_sum_therm / acc_count_therm);

  /* 7. Production Phase */
  double acc_sum_prod = 0.0;
  long acc_count_prod = 0;
  long log_stride = (P.n_sweeps >= 10) ? (P.n_sweeps / 10) : 1;

/* Snapshot Configuration */
#ifdef ENABLE_SNAPSHOTS
  const long N_SNAPSHOTS = 80;
  long snapshot_stride = P.n_sweeps / N_SNAPSHOTS;
  if (snapshot_stride < 1)
    snapshot_stride = 1;
  long next_snapshot = P.n_therm + snapshot_stride;
  int snapshot_idx = 0;
#endif

  for (long sweep = 0; sweep < P.n_sweeps; sweep++) {
    double acc = ising_sweep_metropolis(P.beta, &E, &M);
    acc_sum_prod += acc;
    acc_count_prod++;

    long sweep_MC = P.n_therm + sweep + 1;

    /* Progress Log */
    if ((sweep + 1) % log_stride == 0) {
      double acc_mean = acc_sum_prod / (double)acc_count_prod;
      printf("Prod sweep %ld / %ld | Acc = %.3f\n", sweep + 1, P.n_sweeps,
             acc_mean);
    }

/* Drift Detection (Integrity Check) */
#ifdef MY_DEBUG
    if ((sweep + 1) % 50000 == 0) {
      long E_check = ising_total_energy();
      long M_check = ising_total_magnetization();
      if (E != E_check || M != M_check) {
        fprintf(stderr, "[WARNING] Drift detected at sweep %ld\n", sweep);
        fprintf(stderr, "  Tracked: E=%ld, M=%ld\n", E, M);
        fprintf(stderr, "  Actual:  E=%ld, M=%ld\n", E_check, M_check);
        E = E_check; // Auto-correction
        M = M_check;
      }
    }
#endif

/* Save Snapshots */
#ifdef ENABLE_SNAPSHOTS
    if (sweep_MC >= next_snapshot) {
      save_snapshot(P.L, spin, snapshot_idx, P.beta);
      snapshot_idx++;
      next_snapshot += snapshot_stride;
    }
#endif

    /* Measurements */
    if ((sweep + 1) % P.meas_stride == 0) {
      double e = (double)E / V;
      double m = (double)M / V;
      double e2 = e * e;
      double m2 = m * m;
      double m4 = m2 * m2;

#ifdef USE_BINARY_OUTPUT
      measurement_t meas = {sweep_MC, e, m, e2, m2, m4};
      fwrite(&meas, sizeof(measurement_t), 1, fp);
#else
      fprintf(fp, "%ld %.12f %.12f %.12f %.12f %.12f\n", sweep_MC, e, m, e2, m2,
              m4);
#endif
    }
  }

  fclose(fp);
  lattice_free();
  printf(">> Production Finished.\n");

  return EXIT_SUCCESS;
}

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================
 */

/* Saves lattice configuration to a DAT file in current directory */
void save_snapshot(int L, const signed char *spin, int snapshot_idx,
                   double beta) {
  char fname[256];
  /* Save to snapshots/ subdirectory relative to CWD */
  snprintf(fname, sizeof(fname), "snapshots/conf_L%d_%04d_beta%.3f.dat", L,
           snapshot_idx, beta);

  FILE *fp = fopen(fname, "w");
  if (!fp) {
    /* If open fails (e.g., directory missing), print warning once */
    static int warned = 0;
    if (!warned) {
      perror(
          "[WARN] Failed to save snapshot (check if 'snapshots' dir exists)");
      warned = 1;
    }
    return;
  }

  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      fprintf(fp, "%d ", spin[i * L + j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

/* --- Input Parsing Implementation --- */

int parseInt(const char *arg, const char *name) {
  char *endptr;
  long val = strtol(arg, &endptr, 10);
  if (*endptr != '\0') {
    fprintf(stderr, "[ERROR] Non-numeric characters in <%s>\n", name);
    exit(EXIT_FAILURE);
  }
  if (val <= 0) {
    fprintf(stderr, "[ERROR] Invalid positive value for <%s>\n", name);
    exit(EXIT_FAILURE);
  }
  return (int)val;
}

long parseLong(const char *arg, const char *name) {
  char *endptr;
  long val = strtol(arg, &endptr, 10);
  if (*endptr != '\0') {
    fprintf(stderr, "[ERROR] Non-numeric characters in <%s>\n", name);
    exit(EXIT_FAILURE);
  }
  return val;
}

double parseDouble(const char *arg, const char *name) {
  char *endptr;
  double val = strtod(arg, &endptr);
  if (*endptr != '\0') {
    fprintf(stderr, "[ERROR] Non-numeric characters in <%s>\n", name);
    exit(EXIT_FAILURE);
  }
  if (val <= 0.0 || !isfinite(val)) {
    fprintf(stderr, "[ERROR] Invalid value for <%s>\n", name);
    exit(EXIT_FAILURE);
  }
  return val;
}

static sim_params_t parse_args(int argc, char **argv) {
  if (argc != 7) {
    printf("\nUsage: %s <L> <beta> <n_therm> <n_sweeps> <meas_stride> "
           "<init_type>\n",
           argv[0]);
    exit(EXIT_FAILURE);
  }

  sim_params_t p;
  p.L = parseInt(argv[1], "L");
  p.beta = parseDouble(argv[2], "beta");
  p.n_therm = parseLong(argv[3], "n_therm");
  p.n_sweeps = parseLong(argv[4], "n_sweeps");
  p.meas_stride = parseLong(argv[5], "meas_stride");

  if (strcmp(argv[6], "rng") == 0) {
    strcpy(p.init_type, "rng");
  } else if (strcmp(argv[6], "up") == 0) {
    strcpy(p.init_type, "up");
  } else {
    fprintf(stderr, "[ERROR] init_type must be 'rng' or 'up'\n");
    exit(EXIT_FAILURE);
  }

  if (p.n_sweeps <= 0) {
    fprintf(stderr, "[ERROR] n_sweeps must be > 0\n");
    exit(EXIT_FAILURE);
  }
  if (p.n_therm < 0) {
    fprintf(stderr, "[ERROR] n_therm must be >= 0\n");
    exit(EXIT_FAILURE);
  }
  if (p.meas_stride <= 0 || p.meas_stride > p.n_sweeps) {
    fprintf(stderr, "[ERROR] meas_stride invalid\n");
    exit(EXIT_FAILURE);
  }

  return p;
}
