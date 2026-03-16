#!/bin/bash
#
# Comprehensive evaluation script for ENERO model on single topologies
# Usage: ./eval_all.sh
#
# This script:
# 1. Evaluates the best ENERO model (checkpoint 414) on 3 topologies
# 2. Each topology is evaluated with 50 Traffic Matrices
# 3. Results are stored in evalRes_NEW_* folders
# 4. Generates Figures 5, 6, 7 from evaluation results

set -e

# Configuration
DATASET_BASE="../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1"
LOG_FILE="./Logs/expEnero_3top_15_B_NEWLogs.txt"
MODEL_ID="414"
NUM_PROCESSES="4"

# Topologies to evaluate
TOPOLOGIES=("EliBackbone" "HurricaneElectric" "Janetbackbone")
TOPOLOGY_DIRS=("NEW_EliBackbone" "NEW_HurricaneElectric" "NEW_Janetbackbone")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}ENERO Model Evaluation Pipeline${NC}"
echo -e "${YELLOW}================================================${NC}"

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${RED}Error: Log file $LOG_FILE not found!${NC}"
    exit 1
fi

# Extract differentiation string from log file
DIFF_STR=$(grep -o "Enero_3top_15_B_NEW" "$LOG_FILE" | head -1)
if [ -z "$DIFF_STR" ]; then
    DIFF_STR="Enero_3top_15_B_NEW"
fi

echo -e "${GREEN}Using model: $DIFF_STR with checkpoint $MODEL_ID${NC}"
echo -e "${GREEN}Dataset base: $DATASET_BASE${NC}"
echo -e ""

# ============================================
# STEP 1: Evaluate on Single Topologies
# ============================================
echo -e "${YELLOW}[STEP 1] Evaluating on Single Topologies...${NC}"
echo -e "${YELLOW}This will take a while (~30-60 minutes depending on parallelization)${NC}"

for i in {0..2}; do
    TOPO="${TOPOLOGIES[$i]}"
    TOPO_DIR="${TOPOLOGY_DIRS[$i]}"
    
    echo -e "${GREEN}Evaluating topology: $TOPO${NC}"
    
    python3 eval_on_single_topology.py \
        -d "$LOG_FILE" \
        -f1 results_my_3_tops_unif_05-1 \
        -f2 "$TOPO_DIR/EVALUATE" \
        -max_edge 100 \
        -min_edge 5 \
        -max_nodes 30 \
        -min_nodes 1 \
        -n "$NUM_PROCESSES"
    
    echo -e "${GREEN}✓ Completed evaluation for $TOPO${NC}"
done

echo -e "${GREEN}✓ All single topology evaluations completed!${NC}"
echo -e ""

# ============================================
# STEP 2: Verify results were saved
# ============================================
echo -e "${YELLOW}[STEP 2] Verifying evaluation results...${NC}"

RESULTS_OK=true
for i in {0..2}; do
    TOPO="${TOPOLOGIES[$i]}"
    TOPO_DIR="${TOPOLOGY_DIRS[$i]}"
    
    RESULT_DIR="$DATASET_BASE/evalRes_${TOPO_DIR}/EVALUATE/${DIFF_STR}/${TOPO}/"
    COUNT=$(find "$RESULT_DIR" -name "*.timesteps" 2>/dev/null | wc -l)
    
    if [ "$COUNT" -ge 40 ]; then
        echo -e "${GREEN}✓ $TOPO: $COUNT timesteps files found (Expected ~50)${NC}"
    else
        echo -e "${RED}✗ $TOPO: Only $COUNT timesteps files found (Expected ~50)${NC}"
        RESULTS_OK=false
    fi
done

if ! $RESULTS_OK; then
    echo -e "${RED}Warning: Some results may be incomplete. Check output folders.${NC}"
fi

echo -e "${GREEN}✓ Results verification completed!${NC}"
echo -e ""

# ============================================
# STEP 3: Generate Figures 5 & 6
# ============================================
echo -e "${YELLOW}[STEP 3] Generating Figures 5 & 6...${NC}"

python3 figures_5_and_6.py -d "$DIFF_STR"

echo -e "${GREEN}✓ Figures 5 & 6 generated!${NC}"
echo -e ""

# ============================================
# STEP 4: Generate Figure 7 (per topology)
# ============================================
echo -e "${YELLOW}[STEP 4] Generating Figure 7 (per topology)...${NC}"

for i in {0..2}; do
    TOPO="${TOPOLOGIES[$i]}"
    TOPO_DIR="${TOPOLOGY_DIRS[$i]}"
    EVAL_PATH="$DATASET_BASE/evalRes_${TOPO_DIR}/EVALUATE/"
    
    echo -e "${GREEN}Generating Figure 7 for $TOPO${NC}"
    
    python3 figure_7.py \
        -d "$DIFF_STR" \
        -p "$EVAL_PATH" \
        -t "$TOPO"
done

echo -e "${GREEN}✓ Figure 7 generated for all topologies!${NC}"
echo -e ""

# ============================================
# Summary
# ============================================
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✓ EVALUATION PIPELINE COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e ""
echo -e "Generated figures are located in:"
echo -e "  ./Images/EVALUATION/$DIFF_STR/"
echo -e ""
echo -e "Evaluation results are stored in:"
for i in {0..2}; do
    TOPO_DIR="${TOPOLOGY_DIRS[$i]}"
    echo -e "  $DATASET_BASE/evalRes_${TOPO_DIR}/EVALUATE/"
done
