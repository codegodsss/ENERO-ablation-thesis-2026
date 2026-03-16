#!/bin/bash
#
# cuDNN v8.9.7 Installation Script (Deb Package)
# File: ~/Downloads/cudnn-local-repo-ubuntu2004-8.9.7.29_1.0-1_amd64.deb
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}cuDNN v8.9.7 Installation (Ubuntu 20.04 Deb)${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

CUDNN_DEB="$HOME/Downloads/cudnn-local-repo-ubuntu2004-8.9.7.29_1.0-1_amd64.deb"

if [ ! -f "$CUDNN_DEB" ]; then
    echo -e "${RED}Error: cuDNN .deb file not found at $CUDNN_DEB${NC}"
    exit 1
fi

echo -e "${GREEN}Found cuDNN: $CUDNN_DEB${NC}"
echo -e "${GREEN}Size: $(ls -lh $CUDNN_DEB | awk '{print $5}')${NC}"
echo ""

# ============================================
# Step 1: Install .deb package
# ============================================
echo -e "${YELLOW}[Step 1/4] Installing cuDNN .deb package...${NC}"
sudo dpkg -i "$CUDNN_DEB"
echo -e "${GREEN}✓ .deb package installed${NC}"
echo ""

# ============================================
# Step 2: Update package index
# ============================================
echo -e "${YELLOW}[Step 2/4] Updating package repository...${NC}"
sudo apt-get update
echo -e "${GREEN}✓ Package index updated${NC}"
echo ""

# ============================================
# Step 3: Install cuDNN libraries
# ============================================
echo -e "${YELLOW}[Step 3/4] Installing cuDNN libraries and headers...${NC}"
sudo apt-get install -y libcudnn8 libcudnn8-dev libcudnn8-samples
echo -e "${GREEN}✓ cuDNN libraries installed${NC}"
echo ""

# ============================================
# Step 4: Verify installation
# ============================================
echo -e "${YELLOW}[Step 4/4] Verifying installation...${NC}"

if ldconfig -p | grep -q libcudnn.so; then
    CUDNN_LIB=$(ldconfig -p | grep libcudnn.so | head -1 | awk '{print $1}')
    echo -e "${GREEN}✓ cuDNN library found: $CUDNN_LIB${NC}"
else
    echo -e "${RED}✗ cuDNN library not detected${NC}"
fi

if [ -f "/usr/include/cudnn.h" ]; then
    echo -e "${GREEN}✓ cuDNN headers found: /usr/include/cudnn.h${NC}"
else
    echo -e "${RED}✗ cuDNN headers not found${NC}"
fi

echo ""

# ============================================
# Summary
# ============================================
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✓ cuDNN v8.9.7 Installation Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

echo -e "${YELLOW}Next step - Update LD_LIBRARY_PATH (if needed):${NC}"
echo ""
echo "  export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
echo ""
echo "Or add to ~/.bashrc for persistence:"
echo "  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""

echo -e "${YELLOW}Verify TensorFlow GPU support:${NC}"
echo ""
echo "  python3 -c \"import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)\""
echo ""
