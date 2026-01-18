/**
 * @file pcg32.c
 * @brief Minimal implementation of the PCG32 Random Number Generator.
 *
 * PCG32 (Permuted Congruential Generator) offers excellent statistical properties,
 * small state size, and high performance compared to standard LCGs.
 *
 * Algorithm and implementation based on M.E. O'Neill / pcg-random.org
 * Licensed under Apache License 2.0.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Project headers */
#include "pcg32.h"

/* ============================================================================
 * CORE GENERATOR LOGIC
 * ============================================================================ */

/**
 * @brief Generates the next 32-bit random number.
 * * Uses the LCG state transition followed by the XSH-RR (xorshift high, 
 * random rotation) output permutation function.
 * * @param rng Pointer to the generator state.
 * @return Uniformly distributed uint32_t.
 */
uint32_t pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    
    /* Advance internal state: state = state * multiplier + increment */
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    
    /* Calculate output function (XSH RR) */
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    
    /* Apply random rotation */
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
 * @brief Initializes the PCG32 generator.
 * * @param rng Pointer to the state to initialize.
 * @param initstate Initial state (seed 1).
 * @param initseq Sequence selector (seed 2), selects one of 2^63 streams.
 */
void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq)
{
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;  /* Controls the stream */
    
    pcg32_random_r(rng);              /* Warm-up step 1 */
    rng->state += initstate;
    pcg32_random_r(rng);              /* Warm-up step 2 */
}


/* ============================================================================
 * GLOBAL STATE WRAPPERS
 * ============================================================================ */

/* Global instance used by myrand() */
pcg32_random_t pcg32_random_state;

/**
 * @brief Initializes the global random number generator.
 * Wrapper around pcg32_srandom_r for the global state.
 */
void myrand_init(unsigned long int initstate, unsigned long int initseq)
{
    pcg32_srandom_r(&pcg32_random_state, (uint64_t)initstate, (uint64_t)initseq);
}

/**
 * @brief Generates a double-precision float in [0, 1).
 * * Converts the 32-bit integer output to a double.
 * Division by (UINT32_MAX + 1.0) ensures the range is strictly less than 1.0.
 */
double myrand(void)
{
    return (double)pcg32_random_r(&pcg32_random_state) / ((double)UINT32_MAX + 1.0);
}
