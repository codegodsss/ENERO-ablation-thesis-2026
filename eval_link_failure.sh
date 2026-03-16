#!/bin/bash
#
# Link Failure Evaluation Script for ENERO Model
# Usage: ./eval_link_failure.sh
#
# This script:
# 1. Evaluates ENERO model on topologies with simulated link failures
# 2. Tests model robustness under failure scenarios
# 3. Generates Figure 8 with comparison to baselines

set -e

# Configuration
DATASET_LINK_FAILURE="../Enero_datasets/link_failure_topologies"
LOG_FILE="./Logs/expEnero_3top_15_B_NEWLogs.txt"
MODEL_ID="414"
NUM_PROCESSES="4"
NUM_TOPOLOGIES="20"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}ENERO Link Failure Evaluation${NC}"
echo -e "${YELLOW}================================================${NC}"

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${RED}Error: Log file $LOG_FILE not found!${NC}"
    exit 1
fi

# Extract differentiation string
DIFF_STR=$(grep -o "Enero_3top_15_B_NEW" "$LOG_FILE" | head -1)
if [ -z "$DIFF_STR" ]; then
    DIFF_STR="Enero_3top_15_B_NEW"
fi

echo -e "${GREEN}Using model: $DIFF_STR with checkpoint $MODEL_ID${NC}"
echo -e "${GREEN}Evaluating on $NUM_TOPOLOGIES link failure scenarios${NC}"
echo -e ""

# ============================================
# Check if link failure dataset exists
# ============================================
if [ ! -d "$DATASET_LINK_FAILURE" ]; then
    echo -e "${RED}Error: Link failure dataset directory not found at $DATASET_LINK_FAILURE${NC}"
    echo -e "${YELLOW}Please ensure link failure topologies are pre-generated${NC}"
    exit 1
fi

# ============================================
# STEP 1: Evaluate on Link Failure Scenarios
# ============================================
echo -e "${YELLOW}[STEP 1] Evaluating on Link Failure Scenarios...${NC}"
echo -e "${YELLOW}This will take ~20-30 minutes${NC}"

python3 eval_on_link_failure_topologies.py \
    -d "$LOG_FILE" \
    -f "$DATASET_LINK_FAILURE" \
    -max_edge 100 \
    -min_edge 5 \
    -max_nodes 30 \
    -min_nodes 1 \
    -n "$NUM_PROCESSES"

echo -e "${GREEN}✓ Link failure evaluation completed!${NC}"
echo -e ""

# ============================================
# STEP 2: Generate Figure 8
# ============================================
echo -e "${YELLOW}[STEP 2] Generating Figure 8 (Link Failure Analysis)...${NC}"

LINK_FAILURE_PATH="$DATASET_LINK_FAILURE/evalRes_NEW_$(basename $DATASET_LINK_FAILURE)/EVALUATE/"

python3 figure_8.py \
    -d "$DIFF_STR" \
    -num_topologies "$NUM_TOPOLOGIES" \
    -f "$LINK_FAILURE_PATH"

echo -e "${GREEN}✓ Figure 8 generated!${NC}"
echo -e ""

# ============================================
# Summary
# ============================================
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✓ LINK FAILURE EVALUATION COMPLETED!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e ""
echo -e "Figure 8 location:"
echo -e "  ./Images/EVALUATION/$DIFF_STR/"
echo -e ""
echo -e "Results location:"
echo -e "  $LINK_FAILURE_PATH"
