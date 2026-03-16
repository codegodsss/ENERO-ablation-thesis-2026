#!/bin/bash
#
# Dependency Check Script for ENERO TensorFlow 2.11.0 Setup
# Usage: ./check_dependencies.sh
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}ENERO Model - Dependency Check${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

MISSING_DEPS=0

# ============================================
# 1. Check NVIDIA Driver
# ============================================
echo -e "${YELLOW}[1/6] Checking NVIDIA GPU Driver...${NC}"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ Driver installed: ${DRIVER_VERSION}${NC}"
    echo -e "${GREEN}✓ GPU detected: ${GPU_NAME}${NC}"
else
    echo -e "${RED}✗ NVIDIA GPU Driver NOT found${NC}"
    echo "  Install from: https://www.nvidia.com/Download/driverDetails.aspx"
    MISSING_DEPS=$((MISSING_DEPS+1))
fi
echo ""

# ============================================
# 2. Check CUDA Toolkit
# ============================================
echo -e "${YELLOW}[2/6] Checking CUDA Toolkit...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep 'release' | awk '{print $5}')
    CUDA_PATH=$(which nvcc | sed 's/\/bin\/nvcc//')
    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        echo -e "${GREEN}✓ CUDA Toolkit: ${CUDA_VERSION}${NC}"
        echo -e "${GREEN}✓ CUDA Path: ${CUDA_PATH}${NC}"
    else
        echo -e "${YELLOW}⚠ CUDA Version: ${CUDA_VERSION} (Expected 11.8)${NC}"
        echo "  Consider updating to CUDA 11.8 for best compatibility"
    fi
else
    echo -e "${RED}✗ CUDA Toolkit NOT found${NC}"
    echo "  Install from: https://developer.nvidia.com/cuda-11-8-0-download-archive"
    MISSING_DEPS=$((MISSING_DEPS+1))
fi
echo ""

# ============================================
# 3. Check cuDNN
# ============================================
echo -e "${YELLOW}[3/6] Checking cuDNN...${NC}"
if ldconfig -p | grep -q libcudnn.so; then
    CUDNN_VERSION=$(ldconfig -p | grep libcudnn.so | head -1 | awk '{print $1}')
    echo -e "${GREEN}✓ cuDNN found: ${CUDNN_VERSION}${NC}"
    ls -lh /usr/local/cuda-11.8/lib64/libcudnn.so* 2>/dev/null | head -1 | awk '{print "  Version:", $9}' || true
else
    echo -e "${RED}✗ cuDNN NOT installed${NC}"
    echo "  Required for GPU acceleration"
    echo "  Run: ./install_cudnn.sh"
    echo "  Or read: CUDA_CUDNN_INSTALLATION.md"
    MISSING_DEPS=$((MISSING_DEPS+1))
fi
echo ""

# ============================================
# 4. Check Python & TensorFlow
# ============================================
echo -e "${YELLOW}[4/6] Checking Python & TensorFlow...${NC}"
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python: ${PY_VERSION}${NC}"
    
    # Check if TensorFlow is installed
    if python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null; then
        TF_VERSION=$(python3 -c "import tensorflow as tf; print(tf.__version__)")
        if [[ "$TF_VERSION" == "2.11"* ]]; then
            echo -e "${GREEN}✓ TensorFlow: ${TF_VERSION}${NC}"
        else
            echo -e "${YELLOW}⚠ TensorFlow: ${TF_VERSION} (Expected 2.11.x)${NC}"
            echo "  Update with: pip install --upgrade -r requirements.txt"
        fi
    else
        echo -e "${RED}✗ TensorFlow NOT installed${NC}"
        echo "  Install with: pip install -r requirements.txt"
        MISSING_DEPS=$((MISSING_DEPS+1))
    fi
else
    echo -e "${RED}✗ Python3 NOT found${NC}"
    MISSING_DEPS=$((MISSING_DEPS+1))
fi
echo ""

# ============================================
# 5. Check Gym & Dependencies
# ============================================
echo -e "${YELLOW}[5/6] Checking Python Dependencies...${NC}"
DEPS=("numpy" "gym" "networkx" "pandas" "scipy" "matplotlib" "seaborn")
MISSING_PYTHON_DEPS=0

for dep in "${DEPS[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        VERSION=$(python3 -c "import $dep; print(getattr($dep, '__version__', 'unknown'))" 2>/dev/null || echo "installed")
        echo -e "${GREEN}✓ ${dep}: ${VERSION}${NC}"
    else
        echo -e "${RED}✗ ${dep} NOT installed${NC}"
        MISSING_PYTHON_DEPS=$((MISSING_PYTHON_DEPS+1))
    fi
done

if [ $MISSING_PYTHON_DEPS -gt 0 ]; then
    MISSING_DEPS=$((MISSING_DEPS+1))
    echo -e "${YELLOW}  Install missing packages: pip install -r requirements.txt${NC}"
fi
echo ""

# ============================================
# 6. Check Model Checkpoints
# ============================================
echo -e "${YELLOW}[6/6] Checking Model Checkpoints...${NC}"
if [ -d "modelsEnero_3top_15_B_NEW" ]; then
    CKPT_COUNT=$(find modelsEnero_3top_15_B_NEW -name "ckpt_ACT-*" | wc -l)
    echo -e "${GREEN}✓ Model directory found${NC}"
    echo -e "${GREEN}✓ Checkpoints: ${CKPT_COUNT}${NC}"
    
    # Check for checkpoint 414 specifically
    if find modelsEnero_3top_15_B_NEW -name "*414*" 2>/dev/null | grep -q .; then
        echo -e "${GREEN}✓ Checkpoint 414 (best model) found${NC}"
    else
        echo -e "${YELLOW}⚠ Checkpoint 414 not found (may be in different checkpoint number)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Model directory not found${NC}"
    echo "  Required for evaluation"
fi
echo ""

# ============================================
# Summary
# ============================================
echo -e "${BLUE}================================================${NC}"
if [ $MISSING_DEPS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL DEPENDENCIES SATISFIED${NC}"
    echo ""
    echo -e "${GREEN}System is ready for training/evaluation!${NC}"
else
    echo -e "${RED}⚠ ${MISSING_DEPS} MISSING DEPENDENCIES${NC}"
    echo ""
    echo -e "${YELLOW}Action items:${NC}"
    echo "  1. Install cuDNN: ./install_cudnn.sh"
    echo "  2. Update Python deps: pip install --upgrade -r requirements.txt"
    echo "  3. Re-run this check: ./check_dependencies.sh"
fi
echo -e "${BLUE}================================================${NC}"
echo ""

# Return appropriate exit code
if [ $MISSING_DEPS -gt 0 ]; then
    exit 1
else
    exit 0
fi
