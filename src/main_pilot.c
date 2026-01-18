/**
 * @file main_pilot.c
 * @brief Pilot simulation for the 2D Ising Model (Calibration Phase).
 *
 * This program performs short Metropolis Monte Carlo simulations to estimate:
 * 1. Thermalization time (n_therm)
 * 2. Integrated autocorrelation time (tau_int)
 * 3. Initial system observables
 *
 * Output: Supports both ASCII (.dat) and Binary (.bin) formats.
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

/* * ========== OUTPUT FORMAT CONTROL ==========
 * Controlled via Makefile: gcc -DUSE_BINARY_OUTPUT ...
 */
#ifndef USE_BINARY_OUTPUT
#define USE_BINARY_OUTPUT
#endif
#ifdef USE_BINARY_OUTPUT
#define FILE_EXT "bin"
#define FILE_MODE "wb"
#else
#define FILE_EXT "dat"
#define FILE_MODE "w"
#endif

/* Debugging & Validation Flags */
// #define MY_DEBUG              // Enable internal drift detection (recommended
// for dev) #define THERMALIZATION_TEST   // Enable verbose logging during
// thermalization

/* ============================================================================
 * STRUCTS & PROTOTYPES
 * ============================================================================
 */

typedef struct {
  int L;             // Lattice linear size
  double beta;       // Inverse temperature
  long n_therm;      // Thermalization sweeps
  long n_sweeps;     // Production sweeps
  long meas_stride;  // Measurement interval
  char init_type[5]; // "rng" or "up"
} sim_params_t;

/* Parsing Utilities */
int parseInt(const char *arg, const char *name);
long parseLong(const char *arg, const char *name);
double parseDouble(const char *arg, const char *name);
static sim_params_t parse_args(int argc, char **argv);

/* ============================================================================
 * MAIN FUNCTION
 * ============================================================================
 */

int main(int argc, char **argv) {
  /* 1. Argument Parsing */
  sim_params_t P = parse_args(argc, argv);

  /* 2. RNG Seeding (Hierarchical Strategy) */
  /* Master seeds based on Time + PID to ensure uniqueness across parallel jobs
   */
  uint64_t time_seed = (uint64_t)time(NULL);
  uint64_t pid_seed = (uint64_t)getpid();

  seedgen_init(time_seed, pid_seed);

  unsigned int seed1 = generate_seed();
  unsigned int seed2 = generate_seed();
  myrand_init(seed1, seed2);

  /* 3. Lattice Allocation */
  lattice_alloc(P.L);
  lattice_build_neighbours();

  /* 4. Initialization */
  if (strcmp(P.init_type, "rng") == 0) {
    lattice_init_rng();
  } else {
    lattice_init_up();
  }

  /* 5. Initial Observables */
  long E = ising_total_energy();
  long M = ising_total_magnetization();

  /* 6. Console Banner */
  printf("==================================================\n");
  printf("|       ISING 2D - PILOT SIMULATION              |\n");
  printf("==================================================\n");
  printf(" L             = %d  (V = %d)\n", P.L, V);
  printf(" beta          = %.6f\n", P.beta);
  printf(" n_therm       = %ld\n", P.n_therm);
  printf(" n_sweeps      = %ld\n", P.n_sweeps);
  printf(" meas_stride   = %ld\n", P.meas_stride);
  printf(" init_type     = %s\n", P.init_type);
  printf(" output        = .%s\n", FILE_EXT);
  printf("--------------------------------------------------\n");

  /* 7. Output File Setup */
  /* Note: Filename is relative to CWD. The orchestration script handles
   * directories. */
  char fname[256];
  snprintf(fname, sizeof(fname), "%s_obs_L%d_beta%.6f.%s", P.init_type, P.L,
           P.beta, FILE_EXT);

  FILE *fp = fopen(fname, FILE_MODE);
  if (!fp) {
    perror("[Error] Failed to open output file");
    exit(EXIT_FAILURE);
  }

/* 8. Write Header */
#ifdef USE_BINARY_OUTPUT
  /* Binary Header: Fixed-size struct for portability */
  typedef struct {
    int32_t L;
    double beta;
    int64_t n_therm;
    int64_t n_sweeps;
    int64_t meas_stride;
    uint32_t seed1;
    uint32_t seed2;
  } header_t;

  header_t header = {P.L,           P.beta, P.n_therm, P.n_sweeps,
                     P.meas_stride, seed1,  seed2};
  fwrite(&header, sizeof(header_t), 1, fp);
#else
  /* ASCII Header: Human-readable metadata */
  fprintf(fp,
          "# L=%d beta=%.6f n_therm=%ld n_sweeps=%ld stride=%ld s1=%u s2=%u\n",
          P.L, P.beta, P.n_therm, P.n_sweeps, P.meas_stride, seed1, seed2);
  fprintf(fp, "# sweep e m e2 m2 m4\n");
#endif

  /* * ========================================================================
   * PHASE 1: THERMALIZATION
   * ========================================================================
   */
  double acc_sum_therm = 0.0;
  long acc_count_therm = 0;

  for (long t = 0; t < P.n_therm; t++) {
    double acc = ising_sweep_metropolis(P.beta, &E, &M);
    acc_sum_therm += acc;
    acc_count_therm++;

    /* Console Log (Every 1000 steps) */
    if ((t + 1) % 1000 == 0) {
      double acc_mean = acc_sum_therm / (double)acc_count_therm;
      printf(">> Therm sweep %ld | Mean Acc = %.3f\n", t + 1, acc_mean);
    }

/* Detailed Thermalization Log (Optional) */
#ifdef THERMALIZATION_TEST
    if (t % 100 == 0) {
      char fname_test[256];
      snprintf(fname_test, sizeof(fname_test), "therm_log_L%d_beta%.3f.dat",
               P.L, P.beta);
      FILE *fp_test = fopen(fname_test, "a");
      if (fp_test) {
        double e = (double)E / V;
        double m = (double)M / V;
        fprintf(fp_test, "%ld %.6f %.6f\n", t, e, fabs(m));
        fclose(fp_test);
      }
    }
#endif
  }

  double final_acc_therm = acc_sum_therm / (double)acc_count_therm;
  printf("\n>>> Thermalization Complete. Avg Acc = %.4f\n\n", final_acc_therm);

  /* * ========================================================================
   * PHASE 2: PRODUCTION
   * ========================================================================
   */

  /* Setup variables */
  long log_stride = (P.n_sweeps >= 10) ? (P.n_sweeps / 10) : 1;
  double acc_sum_prod = 0.0;
  long acc_count_prod = 0;

#ifndef THERMALIZATION_TEST
  for (long sweep = 0; sweep < P.n_sweeps; sweep++) {

    /* 1. Metropolis Step */
    double acc = ising_sweep_metropolis(P.beta, &E, &M);
    acc_sum_prod += acc;
    acc_count_prod++;

    long sweep_MC = P.n_therm + sweep + 1; // Absolute sweep index

    /* 2. Progress Logging */
    if ((sweep + 1) % log_stride == 0) {
      double acc_mean = acc_sum_prod / (double)acc_count_prod;
      printf("[Prod] Sweep %ld / %ld | Mean Acc = %.3f\n", sweep + 1,
             P.n_sweeps, acc_mean);
    }

/* 3. Numerical Drift Check (Validation) */
#ifdef MY_DEBUG
    if ((sweep + 1) % 50000 == 0) {
      long E_check = ising_total_energy();
      long M_check = ising_total_magnetization();

      if (E != E_check || M != M_check) {
        fprintf(stderr, "[WARNING] Drift detected at sweep %ld\n", sweep);
        fprintf(stderr, "  Current: E=%ld, M=%ld\n", E, M);
        fprintf(stderr, "  Actual:  E=%ld, M=%ld\n", E_check, M_check);
        // Auto-correction
        E = E_check;
        M = M_check;
      }
    }
#endif

    /* 4. Measurements */
    if ((sweep + 1) % P.meas_stride == 0) {
      double e = (double)E / (double)V;
      double m = (double)M / (double)V;
      double e2 = e * e;
      double m2 = m * m;
      double m4 = m2 * m2;

#ifdef USE_BINARY_OUTPUT
      /* Efficient packed write */
      typedef struct {
        int64_t sweep;
        double e, m, e2, m2, m4;
      } measurement_t;

      measurement_t meas = {sweep_MC, e, m, e2, m2, m4};
      fwrite(&meas, sizeof(measurement_t), 1, fp);
#else
      /* Readable ASCII write */
      fprintf(fp, "%ld %.12f %.12f %.12f %.12f %.12f\n", sweep_MC, e, m, e2, m2,
              m4);
#endif
    }
  }
#endif /* !THERMALIZATION_TEST */

  /* Cleanup */
  fclose(fp);
  lattice_free();

  return EXIT_SUCCESS;
}

/* ============================================================================
 * INPUT PARSING IMPLEMENTATION
 * ============================================================================
 */

int parseInt(const char *arg, const char *name) {
  char *endptr;
  long val = strtol(arg, &endptr, 10);
  if (*endptr != '\0') {
    fprintf(stderr, "[Error] Non-numeric characters in argument <%s>\n", name);
    exit(EXIT_FAILURE);
  }
  if (val <= 0) {
    fprintf(stderr, "[Error] Invalid value for <%s> (must be > 0)\n", name);
    exit(EXIT_FAILURE);
  }
  return (int)val;
}

long parseLong(const char *arg, const char *name) {
  char *endptr;
  long val = strtol(arg, &endptr, 10);
  if (*endptr != '\0') {
    fprintf(stderr, "[Error] Non-numeric characters in argument <%s>\n", name);
    exit(EXIT_FAILURE);
  }
  return val;
}

double parseDouble(const char *arg, const char *name) {
  char *endptr;
  double val = strtod(arg, &endptr);
  if (*endptr != '\0') {
    fprintf(stderr, "[Error] Non-numeric characters in argument <%s>\n", name);
    exit(EXIT_FAILURE);
  }
  if (val <= 0.0 || !isfinite(val)) {
    fprintf(stderr, "[Error] Argument <%s> must be positive and finite\n",
            name);
    exit(EXIT_FAILURE);
  }
  return val;
}

static sim_params_t parse_args(int argc, char **argv) {
  if (argc != 7) {
    printf("\nUsage: %s <L> <beta> <n_therm> <n_sweeps> <meas_stride> "
           "<init_type>\n",
           argv[0]);
    printf("  L            : Lattice size\n");
    printf("  beta         : Inverse temperature\n");
    printf("  n_therm      : Thermalization sweeps\n");
    printf("  n_sweeps     : Production sweeps\n");
    printf("  meas_stride  : Measurement interval\n");
    printf("  init_type    : 'rng' or 'up'\n\n");
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
  } else if (strcmp(argv[6], "up") == 0 ||
             strcmp(argv[6], "cold") ==
                 0) { // Keep "cold" for compat if needed, or strictly "up"
    strcpy(p.init_type, "up");
  } else {
    fprintf(stderr, "[Error] init_type must be 'rng' or 'up'\n");
    exit(EXIT_FAILURE);
  }

  if (p.n_sweeps <= 0) {
    fprintf(stderr, "[Error] n_sweeps > 0 required\n");
    exit(EXIT_FAILURE);
  }
  if (p.n_therm < 0) {
    fprintf(stderr, "[Error] n_therm >= 0 required\n");
    exit(EXIT_FAILURE);
  }
  if (p.meas_stride <= 0 || p.meas_stride > p.n_sweeps) {
    fprintf(stderr, "[Error] meas_stride invalid\n");
    exit(EXIT_FAILURE);
  }

  return p;
}
