#!/bin/bash
#
# NVIDIA CUDA & cuDNN Installation Guide for TensorFlow 2.11.0
# Created: 2026-03-17
# 
# Requirements for TensorFlow 2.11.0:
#   - CUDA Toolkit: 11.8 (✓ Already installed)
#   - cuDNN: 8.6+ (✗ Need to install)
#   - GPU Driver: 570.133.07 (✓ Already installed)
#

set -e

echo "=============================================="
echo "CUDA & cuDNN Installation Guide"
echo "=============================================="
echo ""

# Check current status
echo "📊 Current System Status:"
echo ""
echo "✓ NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "✓ CUDA Toolkit: $(nvcc --version | grep 'release' | awk '{print $5}')"
echo "✗ cuDNN: NOT INSTALLED"
echo ""

echo "=============================================="
echo "STEP 1: Download cuDNN"
echo "=============================================="
echo ""
echo "Visit: https://developer.nvidia.com/cudnn"
echo ""
echo "Instructions:"
echo "  1. Sign in to NVIDIA Developer Account (or create one)"
echo "  2. Go to cuDNN Download page"
echo "  3. Select: cuDNN 8.6.0 for CUDA 11.x"
echo "  4. Choose: Linux x86_64 (tar.xz format)"
echo "  5. Download: cudnn-linux-x86_64-8.6.0.tar.xz"
echo ""

CUDNN_DIR="$HOME/Downloads/cudnn-linux-x86_64-8.6.0.tar.xz"

if [ -f "$CUDNN_DIR" ]; then
    echo "✓ Found cuDNN at: $CUDNN_DIR"
    echo ""
    echo "=============================================="
    echo "STEP 2: Extract cuDNN"
    echo "=============================================="
    echo ""
    
    # Extract to temporary directory
    TEMP_DIR="/tmp/cudnn_extract"
    mkdir -p "$TEMP_DIR"
    
    echo "Extracting cuDNN..."
    tar -xf "$CUDNN_DIR" -C "$TEMP_DIR"
    
    echo "✓ cuDNN extracted"
    echo ""
    
    echo "=============================================="
    echo "STEP 3: Copy cuDNN to CUDA Directory"
    echo "=============================================="
    echo ""
    
    CUDA_PATH="/usr/local/cuda-11.8"
    
    echo "Copying library files to $CUDA_PATH..."
    echo ""
    
    # Copy library files
    sudo cp "$TEMP_DIR/cudnn-linux-x86_64-8.6.0"/lib/*.so* "$CUDA_PATH/lib64/" 2>/dev/null && \
        echo "✓ Library files copied" || \
        echo "Note: Some files may already exist"
    
    # Copy header files
    sudo cp "$TEMP_DIR/cudnn-linux-x86_64-8.6.0"/include/cudnn*.h "$CUDA_PATH/include/" 2>/dev/null && \
        echo "✓ Header files copied" || \
        echo "Note: Some files may already exist"
    
    echo ""
    
    echo "=============================================="
    echo "STEP 4: Set Permissions"
    echo "=============================================="
    echo ""
    
    sudo chmod a+r "$CUDA_PATH/lib64/libcudnn*"
    sudo chmod a+r "$CUDA_PATH/include/cudnn*.h"
    
    echo "✓ Permissions set"
    echo ""
    
    echo "=============================================="
    echo "STEP 5: Verify Installation"
    echo "=============================================="
    echo ""
    
    echo "Checking cuDNN libraries..."
    ls -lh "$CUDA_PATH/lib64"/libcudnn* 2>/dev/null && echo "✓ cuDNN libraries found" || echo "✗ cuDNN libraries not found"
    
    echo ""
    echo "Checking cuDNN headers..."
    ls -lh "$CUDA_PATH/include"/cudnn*.h 2>/dev/null && echo "✓ cuDNN headers found" || echo "✗ cuDNN headers not found"
    
    echo ""
    
    # Cleanup
    rm -rf "$TEMP_DIR"
    echo "✓ Temporary files cleaned up"
    echo ""
    
else
    echo "⚠️  cuDNN file not found at $CUDNN_DIR"
    echo ""
    echo "Please download cuDNN first:"
    echo ""
    echo "  1. Visit: https://developer.nvidia.com/cudnn"
    echo "  2. Select cuDNN 8.6.0 for CUDA 11.x (Linux x86_64)"
    echo "  3. Download to: ~/Downloads/"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "=============================================="
echo "✓ Installation Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Update LD_LIBRARY_PATH in your shell:"
echo ""
echo "     export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "  2. Add to ~/.bashrc for persistence:"
echo ""
echo "     echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"
echo "     source ~/.bashrc"
echo ""
echo "  3. Verify TensorFlow can find cuDNN:"
echo ""
echo "     python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_build_info()[\"cuda_version\"]); print(tf.sysconfig.get_build_info()[\"cudnn_version\"])'"
echo ""
