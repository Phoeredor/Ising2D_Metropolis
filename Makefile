# =============================================================================
#  Ising 2D Metropolis - Professional Makefile
# =============================================================================

# --- Compiler Settings ---
CC      = gcc
# CFLAGS: Optimization, Warnings, Binary Output Macro, Dependency Generation
CFLAGS  = -Wall -Wextra -O3 -march=native -flto -DUSE_BINARY_OUTPUT -MMD -MP
LDFLAGS = -lm

# --- Directories ---
INC_DIR = include
SRC_DIR = src
OBJ_DIR = obj

# --- Executables ---
TARGET_PILOT = pilot_ising
TARGET_PROD  = prod_ising

# --- Sources ---
# Common physics and utility sources
COMMON_SRCS = $(SRC_DIR)/ising.c \
              $(SRC_DIR)/lattice.c \
              $(SRC_DIR)/pcg32.c \
              $(SRC_DIR)/seed_generator.c

# Objects generation (map .c -> .o inside obj/ dir)
COMMON_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(COMMON_SRCS))

# Main objects
PILOT_OBJ   = $(OBJ_DIR)/main_pilot.o
PROD_OBJ    = $(OBJ_DIR)/main_prod.o

# --- Targets ---

.PHONY: all clean directories

all: directories $(TARGET_PILOT) $(TARGET_PROD)

# 1. Link Pilot Executable
$(TARGET_PILOT): $(COMMON_OBJS) $(PILOT_OBJ)
	@echo "[LINK] $@"
	@$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# 2. Link Production Executable
$(TARGET_PROD): $(COMMON_OBJS) $(PROD_OBJ)
	@echo "[LINK] $@"
	@$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# 3. Generic Compilation Rule (for src/*.c -> obj/*.o)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "[CC]   $<"
	@$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

# Create obj directory if not exists
directories:
	@mkdir -p $(OBJ_DIR)

# Clean build artifacts
clean:
	@echo "[CLEAN] Removing executables and objects..."
	@rm -rf $(OBJ_DIR)
	@rm -f $(TARGET_PILOT) $(TARGET_PROD)

# Include automatically generated dependencies (.d files)
-include $(COMMON_OBJS:.o=.d)
-include $(PILOT_OBJ:.o=.d)
-include $(PROD_OBJ:.o=.d)
