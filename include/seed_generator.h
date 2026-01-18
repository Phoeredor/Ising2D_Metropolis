/**
 * @file seed_generator.h
 * @brief Header for the dedicated Seed Generator module.
 *
 * This module wraps a private PCG32 instance to generate high-quality,
 * uncorrelated seeds for initializing parallel simulation threads.
 */

#ifndef SEED_GENERATOR_H
#define SEED_GENERATOR_H

#include <stdint.h>

/* ============================================================================
 * CORE GENERATOR INTERFACE
 * Wrappers around the internal PCG32 state defined in seed_generator.c
 * ============================================================================ */

/**
 * @brief Seeds the internal PCG32 generator.
 * * @param initstate Initial state (64-bit seed).
 * @param initseq Sequence selector (determines the independent stream).
 */
void pcg32_srandom(uint64_t initstate, uint64_t initseq);

/**
 * @brief Generates a 32-bit random number from the internal generator.
 * * @return Uniformly distributed uint32_t.
 */
uint32_t pcg32_random(void);


/* ============================================================================
 * PUBLIC API (User Interface)
 * Functions used by main programs to obtain seeds.
 * ============================================================================ */

/**
 * @brief Initializes the global seed generator.
 * Must be called once at program startup.
 * * @param state Initial state (e.g., time-based).
 * @param seq Sequence number (e.g., PID-based).
 */
void seedgen_init(uint64_t state, uint64_t seq);

/**
 * @brief Generates a fresh random seed.
 * * @return A 32-bit unsigned integer suitable for seeding other RNGs
 * (like the simulation's main pcg32_random_r).
 */
unsigned int generate_seed(void);

/**
 * @brief Debug utility to print a sequence of generated seeds.
 * @param n Number of seeds to generate and print to stdout.
 */
void test_seeds(int n);

#endif /* SEED_GENERATOR_H */
