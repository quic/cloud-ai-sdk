# Introduction 
Developers can access Qualcomm Cloud AI hardware through hyperscalar instances or by purchasing servers equipped with Qualcomm Cloud AI hardware. 

## Hyperscalar Instances 
Hyperscalar instances have all the software tools/packages that a developer needs to run the Cloud AI toolchain. The minimum setup typically includes installation of Apps and Platform SDK. Developers have the ability to launch deep learning containers, load a machine image that fits their needs etc. Details on accessing these instances are coming soon. 

**Developers using hyperscalar instances can skip the rest of the installation section.** 

## On-prem Servers
Developers with on-prem servers need to work with system administators to ensure Cloud AI SDKs are installed and verified properly. It is recommended for developers and system admins to go through the installation section in its entirety. 

# Installation 

The Platform SDK (x86 and Arm) and Apps SDK (x86 only) are targeted for Linux-based platforms. The SDKs can be installed natively on Linux operating systems. Container and orchestration are also supported through Docker and Kubernetes. Virtual machines, including KVM, ESXi, and Hyper-V, are also supported. This section covers:
  - Installation of the SDKs across multiple Linux distributions
  - Building a docker image with the SDKs and third-party packages for a seamless execution of QAic inference tools/workflow
  - Setting up KVM, ESXi, and Hyper-V, and installation of SDKs

## Compilation and Execution modes 
Apps and Platform SDKs enable just-in-time(JIT) or ahead-of-time(AOT) compilation/execution of x86 platforms while only AOT compilation/execution is supported on Arm aarch64. 

In JIT mode, compilation and execution are tightly coupled and require Apps and Platform SDKs to be installed on the same system/VM.

In AOT mode, compilation and execution are decoupled. Networks can be compiled ahead-of-time on x86 (with Apps SDK only) and the compiled networks can be deployed (using Platform SDK) on x86 or Arm aarch64 platforms.

Both JIT and AOT are supported on x86 when Apps and Platform SDK installed on the same server/VM. 

## Supported Operating Systems, Hypervisors, and Platforms 
The AIC100 Platform SDK is compatible with the following operating systems (OS) and platforms.

### Operating Systems

| **Operating systems**        | **Kernel**                          | **X86** | **Arm****™** **(Aarch64)** |
| ---------------------------- | ----------------------------------- | ------- | -------------------------- |
| CentOS Linux 7               | Linux Kernel 5.4.1                  | ✔       | ✗                          |
| CentOS Linux 8               | Linux Kernel 4.19                   | ✗       | ✔                          |
| Ubuntu 18.04                 | Linux Kernel 5.4.1 / Default Kernel | ✔       | ✔ (Note1)                        |
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
| `Note1`: Supported on certain Arm-based Qualcomm platforms. |
| `Note2`: Arm is a trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere. |
| `Note3`: Apps SDK is available only for x86 platforms. |

### Hypervisors
Cloud AI only supports PCIe passthrough to a virtual machine. This means that the virtual machine completely owns the Cloud AI device. A single Cloud AI device cannot be shared between virtual machines or between a virtual machine and the native host. 

| **Hypervisor**                                                                                                                                           | **X86** | **Arm** |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- |
| KVM                                                                                                                                                      | ✔       | ✗       |
| Hyper-V                                                                                                                                                  | ✔       | ✗       |
| ESXi                                                                                                                                                     | ✔       | ✗       |
| `Note` Cloud AI SDKs are not required to be installed on the hypervisor. All the Cloud AI software/SDKs are installed on the guest VMs. |


