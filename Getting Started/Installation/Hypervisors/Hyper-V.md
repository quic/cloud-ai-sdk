# Hyper-V
Hyper-V is a virtualization hypervisor commonly used with Microsoft Windows Server. This enables a native install of Windows Server to act as the host for a virtual machine. The virtual machine emulates a completely independent system that can be limited to a subset of the host hardware, and also can run completely different operating systems from the host.
  
An overview of Hyper-V is at https://en.wikipedia.org/wiki/Hyper-V.

The benefit of a virtual machine is strong isolation. A virtual machine is isolated from other virtual machines and also from the host. This isolation provides a high level of security because one application in a virtual machine may not be aware that it is in a virtual machine, much less that there may be other virtual machines with other applications. Also, a virtual machine provides separation from the host. Even if a driver in the virtual machine crashes the entire virtual machine, that crash will not take down the host and, therefore, will allow other services in other virtual machines to continue operating.

The cost of these benefits is additional overhead to set up the system. Also, additional processing may be required at runtime between the virtual machine and the native hardware.

AIC100 only supports PCIe passthrough to a virtual machine, which means that the virtual machine completely owns the AIC100 device. The AIC100 device cannot be shared between virtual machines, or between a virtual machine and the native host.

The generic setup and operation of a Hyper-V virtual machine is outside the scope of this document. This document assumes that the reader is familiar with those operations and, thus, only explains the elements directly related to assigning an AIC100 device to a Hyper-V virtual machine.

AIC100 supports only the operating systems listed in #FIXME as the operating system running within the virtual machine as the guest OS.

Within a virtual machine, AIC100 still requires the assignment of 32 message signal interrupts (MSI) to operate. Hyper-V does not support emulating an IOMMU in the guest OS of the virtual machine.

However, Hyper-V supports a paravirtualized PCIe root controller that has a driver in the Linux kernel. This driver uses the Hyper-V hypervisor to configure the host IOMMU for the purposes of supplying MSIs for devices like AIC100.

AIC100 requires the following fixes to the Hyper-V PCIe root controller driver within the Linux kernel to operate properly:
■	https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?h=v5.19- rc7&id=08e61e861a0e47e5e1a3fb78406afd6b0cea6b6d
■	https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?h=v5.19- rc7&id=455880dfe292a2bdd3b4ad6a107299fce610e64b
■	https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?h=v5.19- rc7&id=b4b77778ecc5bfbd4e77de1b2fd5c1dd3c655f1f
■	https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?h=v5.19- rc7&id=a2bad844a67b1c7740bda63e87453baf63c3a7f7

Consult the provider of your Linux distribution to confirm that those fixes are present in the Linux kernel used for your Linux distribution.
Once the Hyper-V virtual machine is created, the AIC100 device must be assigned to the virtual machine. Hyper-V calls this direct device assignment (DDA) and it is the same as PCIe passthrough.
Assign the AIC100 device to the Hyper-V virtual machine as follows:
  1.	Configure the virtual machine stop action.
  2.	Disable the device in Windows.
  3.	Unmount the device from Windows.
      NOTE	This step requires the “-force” option to be used. AIC100 currently does not have a partitioning driver. This may be provided in the future.
  4.	Add the device to the virtual machine.

Details on these steps can be found at the following DDA resources:
■	https://docs.microsoft.com/en-us/windows-server/virtualization/hyper-v/deploy/deploying- graphics-devices-using-dda
■	https://docs.microsoft.com/en-us/windows-server/virtualization/hyper-v/deploy/deploying-storage- devices-using-dda
 
To identify the AIC100 device to disable and dismount, look for a device that has the following in the “Hardware Ids” property:
```
PCI\VEN_17CB&DEV_A100&SUBSYS_A10017CB&REV_00
```
Once the AIC100 device is assigned to the virtual machine, it should appear in lspci output in the virtual machine the next time the virtual machine is booted. It will have a different PCIE SBDF address than the host. Install the AIC100 software in the same manner as if you were operating a native system.

