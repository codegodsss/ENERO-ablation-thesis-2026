# ⚡ QUICK ACTION CHECKLIST - CUDA & cuDNN Installation

## Status Summary
- ✅ **NVIDIA GPU Driver**: Installed (570.133.07)
- ✅ **CUDA Toolkit 11.8**: Installed at `/usr/local/cuda-11.8`
- ❌ **cuDNN**: **NOT INSTALLED** ⚠️ ACTION REQUIRED
- ✅ **TensorFlow 2.11.0**: Updated in requirements.txt
- ✅ **Python 3**: Installed

---

## 🎯 ACTION REQUIRED: Install cuDNN

### Option A: Using the Provided Script (EASIEST)

```bash
# 1. Download cuDNN from NVIDIA
#    Go to: https://developer.nvidia.com/cudnn
#    Download: cudnn-linux-x86_64-8.6.0.tar.xz
#    Save to: ~/Downloads/

# 2. Run installation script
cd /home/long/ENERO-ablation-thesis-2026
chmod +x install_cudnn.sh
./install_cudnn.sh

# 3. Set environment variable
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 4. Make it persistent
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Option B: Manual Installation

```bash
# 1. Download from NVIDIA (same as above)

# 2. Extract
tar -xf ~/Downloads/cudnn-linux-x86_64-8.6.0.tar.xz -C ~/Downloads/

# 3. Copy to CUDA
sudo cp ~/Downloads/cudnn-linux-x86_64-8.6.0/lib/*.so* /usr/local/cuda-11.8/lib64/
sudo cp ~/Downloads/cudnn-linux-x86_64-8.6.0/include/cudnn*.h /usr/local/cuda-11.8/include/

# 4. Set permissions
sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h

# 5. Add to PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

---

## ✅ Verification Steps

After installing cuDNN:

```bash
# 1. Check cuDNN files
ls -lh /usr/local/cuda-11.8/lib64/libcudnn.so*
ls -lh /usr/local/cuda-11.8/include/cudnn*.h

# 2. Verify Python can access cuDNN
python3 -c "
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)
"

# 3. Expected output:
# TensorFlow: 2.11.0
# GPU Available: True
```

---

## 📋 Complete Setup Checklist

- [ ] Download cuDNN 8.6.0 for CUDA 11.x
- [ ] Run `install_cudnn.sh` OR manually copy files
- [ ] Add `LD_LIBRARY_PATH` to environment
- [ ] Run verification test
- [ ] Update Python dependencies: `pip install -r requirements.txt`
- [ ] Run dependency check: `./check_dependencies.sh`

---

## 🚀 After cuDNN Installation

Once cuDNN is installed and verified:

```bash
# 1. Update Python dependencies
pip install --upgrade -r requirements.txt

# 2. Run full dependency check
./check_dependencies.sh

# 3. Start evaluation
./eval_all.sh
```

---

## 📞 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot find libcudnn.so" | Run `export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH` |
| Permission denied copying files | Use `sudo cp` or run `install_cudnn.sh` |
| NVIDIA account approval takes time | Wait 24h or use community repository |
| TensorFlow doesn't detect GPU | Verify `LD_LIBRARY_PATH` is set, then restart Python |
| Old TensorFlow installed | Run `pip install --upgrade -r requirements.txt` |

---

## 📚 Reference Documents

- 📄 [CUDA_CUDNN_INSTALLATION.md](CUDA_CUDNN_INSTALLATION.md) - Detailed guide
- 🔧 [install_cudnn.sh](install_cudnn.sh) - Installation script
- ✅ [check_dependencies.sh](check_dependencies.sh) - Dependency checker

---

**Last Updated**: 2026-03-17  
**Status**: Ready for cuDNN installation
