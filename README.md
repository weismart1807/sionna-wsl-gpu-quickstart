# sionna-wsl-gpu-quickstart
#### One-stop guide to install TensorFlow 2.18 GPU and Sionna 1.2.1 on WSL Ubuntu 24.04 using a Python virtual environment.  

### WSL + Ubuntu 24.04 + TF GPU + Sionna
###### (No system CUDA Toolkit/cuDNN required when using pip wheels)

## Prerequisites (GPU drivers) 
##### Windows (WSL): NVIDIA driver >= 528.33
##### Native Linux:  NVIDIA driver >= 525.60.13
######  If these are met, you do NOT need to install CUDA Toolkit 12.3 or cuDNN SDK 8.9.7 system-wide because pip-installed TensorFlow wheels include their own matching CUDA/cuDNN runtime.

## On Windows (PowerShell as Admin)
```bash
wsl --update
wsl --set-default-version 2
wsl --install -d Ubuntu-24.04
```
###### Reboot if prompted, then create your Linux username/password on first launch.

## Inside WSL: Ubuntu 24.04 (bash) 
### 0) Basics
```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install build-essential curl git software-properties-common
```

### 1) Sanity check: GPU is visible in WSL
```bash
nvidia-smi
```

### 2) Python venv
```bash
sudo apt -y install python3-venv python3-pip
python3 -m venv ~/sionna
source ~/sionna/bin/activate
python -V
pip install -U pip
```

### 3) TensorFlow GPU (pip wheel bundles CUDA/cuDNN) + Sionna
```bash
pip install "tensorflow==2.18.1"
pip install "sionna==1.2.1"
```

### 4) Quick GPU test (should list GPU:0 and run matmul on GPU)
```bash
python - << 'PY'
import tensorflow as tf, sys, time
print("TF:", tf.__version__)
print("Py:", sys.version)
print("GPUs:", tf.config.list_physical_devices("GPU"))
with tf.device("/GPU:0"):
    a=tf.random.normal([4096,4096]); b=tf.random.normal([4096,4096])
    t=time.time(); c=tf.matmul(a,b); _=c.numpy()
    print("Device:", c.device, "Elapsed:", time.time()-t, "s")
PY
```

### 5) (Optional) reduce TF startup logs
```bash
echo 'export TF_CPP_MIN_LOG_LEVEL=2' >> ~/.bashrc
source ~/.bashrc
```
