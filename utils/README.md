# Basic Commands/Utilities for Cloud AI 100 devices 

## Create `qaic` group to avoid `sudo` to read card/device status

```
sudo usermod -aG qaic $USER
newgrp qaic
bash
```

## Check device health 
Monitor the health of all AI 100 devices (SoCs) using the `qaic-util` utility. 

```
/opt/qti-aic/tools/qaic-util -q | grep -e Status -e QID
```

## Monitoring of AI 100 devices (SoCs)
Continuously monitor the health, telemetry (temperature, power etc) and resources (compute, DRAM etc) of the AI 100 devices (SoCs) using the `qaic-util` utility. 

```
/opt/qti-aic/tools/qaic-util -t 1
```

## Reset AI 100 devices (SoCs)
To reset **all** AI 100 devices (SoCs), run
```
sudo /opt/qti-aic/tools/qaic-util -s
```

To reset **individual** AI 100 devices (SoCs), run 
```
sudo /opt/qti-aic/tools/qaic-util -s -p <PCIe address SSSS:BB:DD.F>
```
where,
    - SSSS = 4 digits segment number
    - BB = 2 digits bus number
    - DD = 2 digits device number
    - F = 1 digit function number

For example, 
```
sudo /opt/qti-aic/tools/qaic-util -s -p 0000:83:00.0

Resetting 0000:83:00.0:
        0000:83:00.0 success
```