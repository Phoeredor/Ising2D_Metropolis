/**
 * @file seed_generator.c
 * @brief PCG32-based Random Number Generator for seeding purposes.
 *
 * Implements the PCG XSH-RR 64/32 algorithm (http://www.pcg-random.org).
 * Used specifically to generate high-entropy seeds for the main simulation RNG.
 */

#include <stdint.h>
#include <stdio.h>

/* Project headers */
#include "seed_generator.h"
#include "pcg32.h"

/* ============================================================================
 * INTERNAL STATE
 * ============================================================================ */

/**
 * @brief Global internal state for the seed generator.
 * Initialized with arbitrary constants. Should be re-seeded via seedgen_init().
 */
static pcg32_random_t rng_state = { 
    0x853c49e6748fea9bULL,  /* Initial state */
    0xda3e39cb94b95bdbULL   /* Initial increment */
};


/* ============================================================================
 * CORE IMPLEMENTATION
 * ============================================================================ */

/**
 * @brief Generates a 32-bit random number (PCG XSH-RR).
 * @return Uniformly distributed uint32_t.
 */
uint32_t pcg32_random(void) 
{
    uint64_t oldstate = rng_state.state;
    
    /* Linear Congruential Generator step */
    /* Multiplier: 6364136223846793005ULL */
    rng_state.state = oldstate * 6364136223846793005ULL + rng_state.inc;
    
    /* Output function: XSH-RR (XorShift High, Random Rotation) */
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
 * @brief Seeds the PCG32 generator.
 * @param initstate Initial state.
 * @param initseq Sequence selector (stream ID).
 */
void pcg32_srandom(uint64_t initstate, uint64_t initseq) 
{
    rng_state.state = 0U;
    rng_state.inc = (initseq << 1u) | 1u;  /* Ensure increment is odd */
    
    pcg32_random();             /* Warm-up step 1 */
    rng_state.state += initstate;
    pcg32_random();             /* Warm-up step 2 */
}


/* ============================================================================
 * PUBLIC API
 * ============================================================================ */

/**
 * @brief Initializes the seed generator.
 * @param state Initial state (e.g., time based).
 * @param seq Sequence number (e.g., PID based).
 */
void seedgen_init(uint64_t state, uint64_t seq) 
{
    pcg32_srandom(state, seq);
}

/**
 * @brief Returns a fresh random seed.
 * @return A 32-bit random integer suitable for seeding other RNGs.
 */
unsigned int generate_seed(void) 
{
    return pcg32_random();
}

/**
 * @brief Debug utility to print generated seeds.
 */
void test_seeds(int n) 
{
    for (int i = 0; i < n; ++i) {
        printf("Seed %d: %u\n", i + 1, generate_seed());
    }
}
