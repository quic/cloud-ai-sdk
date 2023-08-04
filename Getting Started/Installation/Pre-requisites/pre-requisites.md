# Pre-requisites for Ubuntu 20.04 / 22.04 on x86

- Ubuntu 20.04 / 22.04 with default kernel installed, with minimum options.
- Install the “dkms” package for the kernel.
```
  sudo apt-get install dkms
 ```
- Install the "linux-headers" package for the kernel.
-	Install the following packages:
```
  sudo apt-get update && sudo apt-get install -y software-properties-common sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
  sudo apt-get update && sudo apt-get install -y build-essential git vim libpci-dev libudev-dev python3-pip python3-setuptools python3-wheel python3.8 python3.8-dev  python3.8-venv
  sudo apt-get update && sudo apt-get install -y unzip wget ca-certificates sudo pciutils libglib2.0-dev libssl-dev snap snapd libgl1-mesa-glx openssh- server pkg-config clang-format libpng-dev
  sudo apt-get install libstdc++6 sudo apt-get install libncurses5 sudo pip3 install --upgrade pip
  sudo pip3 install wheel numpy opencv-python onnx
```
- Add/update environment variables:
```
export LD_LIBRARY_PATH=“$LD_LIBRARY_PATH:/opt/qti-aic/dev/lib/x86_64” export PATH="/usr/local/bin:$PATH"
export PATH="$PATH:/opt/qti-aic/tools:/opt/qti-aic/exec:/opt/qti-aic/ scripts"
export QRAN_EXAMPLES="/opt/qti-aic/examples"
export PYTHONPATH="$PYTHONPATH:/opt/qti-aic/dev/lib/x86_64" export QAIC_APPS="/opt/qti-aic/examples/apps"
export QAIC_LIB="/opt/qti-aic/dev/lib/x86_64/libQAic.so"
export QAIC_COMPILER_LIB="/opt/qti-aic/dev/lib/x86_64/libQAicCompiler.so"
```

# Pre-requisites for Ubuntu 18.04 on x86
- Install the “dkms” package and the “linux-headers” package for the kernel.
```
sudo apt-get install dkms
```
- The linux-headers package is already installed for the stock kernel, but for the 5.4.1 kernel, the
headers package needs to be manually installed.
  - Download kernel generic .deb files from https://ci.linaro.org/job/lt-qcom-linux-aic100-bionic/ lastSuccessfulBuild/artifact/out/ and install as follows:
    ```
    sudo dpkg -i linux-headers-5.4.1-050401-generic_5.4.1-050401- generic-1_amd64.deb
    sudo dpkg -i linux-image-5.4.1-050401-generic_5.4.1-050401- generic-1_amd64.deb
    ```
    
- Install the following packages:
```
  sudo apt-get update && sudo apt-get install -y software-properties-common sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
  sudo apt-get update && sudo apt-get install -y build-essential git vim libpci-dev libudev-dev python3-pip python3-setuptools python3-wheel python3.8 python3.8-dev python3.8-venv
  sudo apt-get update && sudo apt-get install -y unzip wget ca-certificates sudo pciutils libglib2.0-dev libssl-dev snap snapd libgl1-mesa-glx openssh- server pkg-config clang-format libpng-dev
  sudo apt-get install -y libstdc++6 sudo apt-get install -y libncurses5 sudo pip3 install --upgrade pip
  sudo pip3 install wheel numpy opencv-python onnx
 ```
- Add/update environment variables:
```
  export LD_LIBRARY_PATH=“$LD_LIBRARY_PATH:/opt/qti-aic/dev/lib/x86_64” export PATH="/usr/local/bin:$PATH"
  export PATH="$PATH:/opt/qti-aic/tools:/opt/qti-aic/exec:/opt/qti-aic/ scripts"
  export QRAN_EXAMPLES="/opt/qti-aic/examples"
  export PYTHONPATH="$PYTHONPATH:/opt/qti-aic/dev/lib/x86_64" export QAIC_APPS="/opt/qti-aic/examples/apps"
  export QAIC_LIB="/opt/qti-aic/dev/lib/x86_64/libQAic.so"
  export QAIC_COMPILER_LIB="/opt/qti-aic/dev/lib/x86_64/libQAicCompiler.so"
```

# Pre-requisites for Centos 7 / RHEL on x86

