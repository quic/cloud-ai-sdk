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
  

