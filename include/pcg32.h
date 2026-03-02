/**
 * @file pcg32.h
 * @brief Header for the PCG32 Random Number Generator.
 *
 * Provides the state structure and function prototypes for the
 * Permuted Congruential Generator (PCG-XSH-RR).
 * Reference: https://www.pcg-random.org/
 */

#ifndef PCG32_H
#define PCG32_H

#include <stdint.h>

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * @brief PCG32 Random Number Generator State.
 *
 * Contains the internal state of the generator.
 * To generate multiple independent streams, use different 'inc' values.
 */
typedef struct {
    uint64_t state;  /**< Current internal state (LCG) */
    uint64_t inc;    /**< Stream selector (must be odd) */
} pcg32_random_t;


/* ============================================================================
 * GLOBAL STATE
 * (Defined in pcg32.c, used by high-level wrappers)
 * ============================================================================ */

extern pcg32_random_t pcg32_random_state;


/* ============================================================================
 * CORE FUNCTIONS (Re-entrant)
 * Use these for thread-safe or multi-stream generation.
 * ============================================================================ */

/**
 * @brief Generates the next 32-bit random number (Low-level).
 * Updates the provided state structure.
 * @param rng Pointer to the PCG32 state.
 * @return Random uint32_t.
 */
uint32_t pcg32_random_r(pcg32_random_t *rng);

/**
 * @brief Seeds a specific PCG32 state.
 * @param rng Pointer to the PCG32 state to initialize.
 * @param initstate Initial state (seed 1).
 * @param initseq Sequence selector (seed 2).
 */
void pcg32_srandom_r(pcg32_random_t *rng, uint64_t initstate, uint64_t initseq);


/* ============================================================================
 * HIGH-LEVEL WRAPPERS (Global State)
 * Simplified interface for general simulation use.
 * ============================================================================ */

/**
 * @brief Initializes the global RNG instance.
 * @param initstate Initial seed.
 * @param initseq Sequence/Stream ID.
 */
void myrand_init(unsigned long int initstate, unsigned long int initseq);

/**
 * @brief Generates a double-precision float in [0, 1).
 * Uses the global RNG state.
 * @return Uniform random double in [0, 1).
 */
double myrand(void);

#endif /* PCG32_H */
