# Installation 

The Platform SDK (x86 and Arm) and Apps SDK (x86 only) are targeted for Linux-based platforms. The SDKs can be installed natively on Linux operating systems. Container and orchestration are also supported through Docker and Kubernetes. Virtual machines, including KVM, ESXi, and Hyper-V, are also supported. This section covers:
  - Installation of the SDKs across multiple Linux distributions
  - Building a docker image with the SDKs and third-party packages for a seamless execution of QAic inference tools/workflow
  - Setting up KVM, ESXi, and Hyper-V, and installation of SDKs

## Supported Operating Systems, Hypervisors, and Platforms 
The AIC100 Platform SDK is compatible with the following operating systems (OS) and platforms.

| **Operating systems**        | **Kernel**                          | **X86** | **Arm****™** **(Aarch64)** |
| ---------------------------- | ----------------------------------- | ------- | -------------------------- |
| CentOS Linux 7               | Linux Kernel 5.4.1                  | ✔       | ✗                          |
| CentOS Linux 8               | Linux Kernel 4.19                   | ✗       | ✔                          |
| Ubuntu 18.04                 | Linux Kernel 5.4.1 / Default Kernel | ✔       | ✔^1^                         |
| Ubuntu 20.04                 | Default Kernel                      | ✔       | ✗                          |
| Ubuntu 22.04                 | Default Kernel                      | ✔       | ✗                          |
| Red Hat Enterprise Linux 7.9 | Default Kernel                      | ✔       | ✗                          |
| Red Hat Enterprise Linux 8.3 | Default Kernel                      | ✔       | ✗                          |
| Red Hat Enterprise Linux 8.4 | Default Kernel                      | ✔       | ✗                          |
| Red Hat Enterprise Linux 8.6 | Default Kernel                      | ✔       | ✗                          |
| Red Hat Enterprise Linux 8.7 | Default Kernel                      | ✔       | ✗                          |
| Red Hat Enterprise Linux 9.0 | Default Kernel                      | ✔       | ✗                          |
| Red Hat Enterprise Linux 9.1 | Default Kernel                      | ✔       | ✗                          |
| AWS linux2                                          | Default Kernel | ✔ | ✗ |
| --------------------------------------------------- | -------------- | - | - |
| ^1^Supported on certain Arm-based Qualcomm platforms. |

**NOTE**:Arm is a trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere. <br>
**NOTE**:Apps SDK is available only for x86 platforms.

| **Hypervisor**                                                                                                                                           | **X86** | **Arm** |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- |
| KVM                                                                                                                                                      | ✔       | ✗       |
| Hyper-V                                                                                                                                                  | ✔       | ✗       |
| ESXi                                                                                                                                                     | ✔       | ✗       |
| **NOTE**    No AI 100 related software/SDKs are required to be installed on the hypervisor. All the AI 100 software/SDKs are installed on the guest VMs. |
