# Platform SDK Installation 

- Developers need to login to https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence/cloud-ai-100. First time users need to register. 
- Click on the Software tab. The latest Platform SDKs are available under “Software Release”. The Platform SDK is split into four packages for the different combinations of x86/Arm and deb/rpm: x86-deb, x86-rpm, aarch64-deb, and aarch64-rpm
- Login as root or use ```sudo``` to have the right permissions to complete installation 
- Uninstall any Platform SDK already installed<br> 
  ```
  sudo ./uninstall.sh
  sync
  ```
- Copy the Patform SDK downloaded from the Qualcomm Portal to the host machine:
    - For networked x86 or Arm host:
      - Use scp, rsync, or samba to copy the Platform SDK zip file to the host machine
      - Log-in to the host machine (ssh or local terminal)
      - Unzip the downloaded zip file to a working directory
      - ```cd``` to the working directory
    - For ARM hosts that support Google Android Debug Bridge (ADB):
      - adb push <Platform SDK zip file> /data
      - adb shell
      - cd /data
      - Unzip the downloaded zip file to a working directory
      - cd to the working directory
  
  Note: Extract the downloaded zip file until qaic-platform-sdk-<x86_64/aarch64>-<deb/rpm>-<SDK version>.zip is available. Confirm the architecture and installation linux package format works for your setup. Unzip 
  
- Install Platform SDK 
  The Platform SDK is composed of the following tree structure. Users will see rpm or deb based on the SDK package:
  ```
    ├── common
    │   ├── sectools 
    ├── <architecture - x86_64 or aarch64>  
    │   ├── rpm 
    │   │   ├── rpm 
    │   │   │   ├── qaic-fw-<version>.el7.x86_64.rpm 
    │   │   │   ├── qaic-kmd-<version>.el7.x86_64.rpm 
    │   │   │   └── qaic-rt-<version>.el7.x86_64.rpm   
    │   │   ├── rpm-docker 
    │   │   │   └── qaic-rt-docker-<version>.el7.x86_64.rpm  
    │   │   ├── install.sh 
    │   │   ├── Notice.txt 
    │   │   └── uninstall.sh 
    │   ├── deb
    │   │   ├── deb 
    │   │   │   ├── qaic-fw_<version>.deb 
    │   │   │   ├── qaic-kmd_<version>-devel_amd64.deb 
    │   │   │   └── qaic-rt_<version>_amd64.debm  
    │   │   ├── install.sh 
    │   │   ├── Notice.txt 
    │   │   └── uninstall.sh 
    │   ├── test_suite   
    │   │   ├── pcietool   
    └── └── └── powerstress    
  ```
  - Run the install.sh script as root or with sudo to install with root permissions. Installation may take up to 30 mins depending on the number of Cloud AI cards in the server/VM. Cloud AI cards undergo resets several times during the installation. 
    - cd <architecture>/<deb/rpm> 
    - For Hyrid boot cards (PCIe CEM form factor cards), run ```sudo ./install.sh --auto_upgrade_sbl --ecc enable```
      For VM on ESXi hypervisor, run ```sudo ./install.sh --auto_upgrade_sbl --datapath_polling –-ecc enable```
    - For Flashless boot cards, run ```sudo ./install.sh –-ecc enable```
      For VM on ESXi hypervisor, run ```sudo ./install.sh --datapath_polling –-ecc enable```
  - On successful installation of the platform SDK, the contents are stored in ```/opt/qti-aic```. <br>
      ```dev  drivers  examples  exec  firmware  services  test-data  tools```

- Verify card health 
  - Check card health using the qaic-util utility. The following command should output "Ready" for all cards present in the system.<br>
    ```sudo /opt/qti-aic/tools/qaic-util -q | grep Status```
  - Check other parameters such as Max Link Speed, Device capabilities, etc. by running the qaic-util tool per card in the system.<br>
    ```sudo /opt/qti-aic/tools/qaic-util -q```
  



