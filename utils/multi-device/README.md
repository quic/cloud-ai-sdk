# Multi Device 

This section provides the setup instructions for tensor slicing across multiple AI 100 devices (SoCs and Cards).

## Pre-requisites 

- Server with Platform and APPS SDK versions >= 1.17 installed.
- PCIe switch for inter-card P2P communication
- python3 -m pip install pyudev

## Setup instructions 

### Disable PCIe ACS for P2P communication between cards 

1. Run `QAicChangeAcs.py` without any flags to display a hierarchial view of PCI bridges and AI 100 devices. 

```
$ python3 QAicChangeAcs.py
Found the following AIC100 devices:
Root
----0000:30:01.1   <-- Host system PCIe switch, script will disable ACS here
--------0000:31:00.0     <-- Ultra AI 100 onboard PCIe switch, script will disable ACS here
------------0000:32:03.0
----------------0000:36:00.0 [Qualcomm AIC100]
------------0000:32:02.0
----------------0000:35:00.0 [Qualcomm AIC100]
------------0000:32:00.0
----------------0000:38:00.0 [Qualcomm AIC100]
------------0000:32:01.0
----------------0000:39:00.0 [Qualcomm AIC100]
--------0000:21:00.0    <-- Ultra AI 100 onboard PCIe switch, script will disable ACS here
------------0000:22:00.0
----------------0000:23:00.0 [Qualcomm AIC100]
------------0000:22:02.0
----------------0000:25:00.0 [Qualcomm AIC100]
------------0000:22:01.0
----------------0000:27:00.0 [Qualcomm AIC100]
------------0000:22:03.0
----------------0000:28:00.0 [Qualcomm AIC100]
```

2. Run `QAicChangeAcs.py all` to disable ACS on all the downstream ports (on the PCIe switch) that connect to AI 100 devices as well as PCIe switch downstream ports that connect to the PCIe switch onboard the AI 100 cards. This command will enable P2P between the AI 100 devices (SoCs) on the same card as well card to card. 

3. Users optionally can selectively disable ACS by running `QAicChangeAcs.py <SSSS:BB:DD.F>`, where 
    - SSSS = 4 digits segment number
    - BB = 2 digits bus number
    - DD = 2 digits device number
    - F = 1 digit function number

    of the nearest common ancestor PCI bridge under which ACS needs to be disabled. 
    
    Examples: 

    `$ python3 QAicChangeAcs.py 0000:31:00.0` will disable ACS on the first set of AI 100 devices (0000:36:00.0, 0000:35:00.0, 0000:38:00.0 and 0000:39:00.0). <br> 
    `$ python3 QAicChangeAcs.py 0000:30:01.1` will disable ACS across both the AI 100 Ultra cards as well as the 4 devices in each AI 100 card <br>

4. Above steps need to be repeated on every server power cycle. 


### Enable multi-device partitioning (MDP)

This step is required everytime a new version of the Platform SDK is installed. 

First command starts the qaic-monitor service. Second command enables MDP across all AI 100 devices in the server. Third command resets all the devices. 
``` 
systemd-run --unit=qmonitor-proxy /opt/qti-aic/tools/qaic-monitor-grpc-server
sudo /opt/qti-aic/tools/qaic-monitor-json -i enable_mdp.json
sudo /opt/qti-aic/tools/qaic-util -s
```
