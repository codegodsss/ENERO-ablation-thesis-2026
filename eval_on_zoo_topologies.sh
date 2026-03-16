#!/bin/bash
#
# Zoo Topologies Evaluation Script for ENERO Model  
# Usage: ./eval_on_zoo_topologies.sh
#
# This script:
# 1. Evaluates ENERO model on 100+ unseen topologies from TopologyZoo
# 2. Tests generalization to new network topologies
# 3. Generates Figure 9 with performance distribution

set -e

# Configuration
ZOO_DATASET="../Enero_datasets/topology_zoo"
LOG_FILE="./Logs/expEnero_3top_15_B_NEWLogs.txt"
NUM_PROCESSES="4"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}ENERO Zoo Topologies Evaluation (Generalization Test)${NC}"
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

echo -e "${GREEN}Using model: $DIFF_STR${NC}"
echo -e ""

# ============================================
# Check if zoo dataset exists
# ============================================
if [ ! -d "$ZOO_DATASET" ]; then
    echo -e "${RED}Error: Zoo dataset not found at $ZOO_DATASET${NC}"
    echo -e "${YELLOW}Please download/prepare TopologyZoo datasets first${NC}"
    exit 1
fi

echo -e "${YELLOW}Zoo dataset location: $ZOO_DATASET${NC}"
TOPO_COUNT=$(find "$ZOO_DATASET" -name "*.graph" 2>/dev/null | wc -l)
echo -e "${GREEN}Found $TOPO_COUNT topologies${NC}"
echo -e ""

# ============================================
# STEP 1: Evaluate on Zoo Topologies
# ============================================
echo -e "${YELLOW}[STEP 1] Evaluating on Zoo Topologies...${NC}"
echo -e "${YELLOW}This will take ~60-120 minutes depending on number of topologies${NC}"

python3 eval_on_zoo_topologies.py \
    -d "$LOG_FILE" \
    -max_edge 500 \
    -min_edge 1 \
    -max_nodes 500 \
    -min_nodes 5 \
    -n "$NUM_PROCESSES"

echo -e "${GREEN}✓ Zoo topology evaluation completed!${NC}"
echo -e ""

# ============================================
# STEP 2: Generate Figure 9
# ============================================
echo -e "${YELLOW}[STEP 2] Generating Figure 9 (Zoo Performance Distribution)...${NC}"

ZOO_RESULTS_PATH="$ZOO_DATASET/evalRes/"

python3 figure_9.py \
    -d "$DIFF_STR" \
    -p "$ZOO_RESULTS_PATH"

echo -e "${GREEN}✓ Figure 9 generated!${NC}"
echo -e ""

# ============================================
# Summary
# ============================================
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✓ ZOO TOPOLOGY EVALUATION COMPLETED!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e ""
echo -e "Figure 9 location:"
echo -e "  ./Images/EVALUATION/$DIFF_STR/"
echo -e ""
echo -e "Results location:"
echo -e "  $ZOO_RESULTS_PATH"
