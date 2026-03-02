# =============================================================================
#  Ising 2D Metropolis - Professional Makefile
# =============================================================================

# --- Compiler Settings ---
CC      = gcc
CFLAGS  = -Wall -Wextra -O3 -march=native -flto -DUSE_BINARY_OUTPUT -MMD -MP
LDFLAGS = -lm

# --- Directories ---
INC_DIR = include
SRC_DIR = src
OBJ_DIR = obj

# --- Executables ---
TARGET_PILOT  = pilot_ising
TARGET_PROD   = prod_ising
TARGET_VISUAL = visual_ising

# --- Sources ---
COMMON_SRCS = $(SRC_DIR)/ising.c \
              $(SRC_DIR)/lattice.c \
              $(SRC_DIR)/pcg32.c \
              $(SRC_DIR)/seed_generator.c

COMMON_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(COMMON_SRCS))

PILOT_OBJ  = $(OBJ_DIR)/main_pilot.o
PROD_OBJ   = $(OBJ_DIR)/main_prod.o

VISUAL_SRCS = $(SRC_DIR)/main_visual.c \
              $(SRC_DIR)/ising.c \
              $(SRC_DIR)/lattice.c \
              $(SRC_DIR)/pcg32.c \
              $(SRC_DIR)/seed_generator.c

# =============================================================================
# --- Targets ---
# =============================================================================

.PHONY: all clean directories

# ← default target: MUST be first
all: directories $(TARGET_PILOT) $(TARGET_PROD)

# 1. Pilot
$(TARGET_PILOT): $(COMMON_OBJS) $(PILOT_OBJ)
	@echo "[LINK] $@"
	@$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# 2. Production
$(TARGET_PROD): $(COMMON_OBJS) $(PROD_OBJ)
	@echo "[LINK] $@"
	@$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# 3. Visual sweep (compiled directly, no obj/ intermediates needed)
$(TARGET_VISUAL): $(VISUAL_SRCS)
	@echo "[LINK] $@"
	@$(CC) $(CFLAGS) -I$(INC_DIR) $(VISUAL_SRCS) -o $@ $(LDFLAGS)

# 4. Generic compilation rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "[CC]   $<"
	@$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

# Create obj/ directory
directories:
	@mkdir -p $(OBJ_DIR)

# Clean everything
clean:
	@echo "[CLEAN] Removing executables and objects..."
	@rm -rf $(OBJ_DIR)
	@rm -f $(TARGET_PILOT) $(TARGET_PROD) $(TARGET_VISUAL)

# Auto-generated dependencies
-include $(COMMON_OBJS:.o=.d)
-include $(PILOT_OBJ:.o=.d)
-include $(PROD_OBJ:.o=.d)

