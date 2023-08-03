Kernel-based Virtual Machine (KVM) is a module in the Linux kernel that allows the Linux OS to operate as a hypervisor. This enables a native install of Linux to act as the host for a virtual machine. In combination with QEMU and libVirt, a KVM virtual machine emulates a completely independent system that can be limited to a subset of the host hardware, and that can also run completely different operating systems from the host.

An overview of KVM is at https://en.wikipedia.org/wiki/Kernel-based_Virtual_Machine.

The benefit of a virtual machine is strong isolation. A virtual machine is isolated from other virtual machines and from the host. This isolation provides a high level of security because one application in a virtual machine may not be aware that it is in a virtual machine, much less that there may be other virtual machines with other applications. Also, a virtual machine provides separation from the host.

Even if a driver in the virtual machine crashes the entire virtual machine, that crash will not take down the host and, therefore, will allow other services in other virtual machines to continue operating.

The cost of these benefits is additional overhead to set up the system. Also, additional processing may be required at runtime between the virtual machine and the native hardware.

AIC100 only supports PCIe passthrough to a virtual machine. This means that the virtual machine completely owns the AIC100 device. The AIC100 device cannot be shared between virtual machines, or between a virtual machine and the native host.

The generic setup and operation of a KVM virtual machine is outside the scope of this document. This document assumes that the reader is familiar with those operations and, thus, only explains the elements directly related to assigning an AIC100 device to a KVM virtual machine.

AIC100 supports only the operating systems listed in #FIXME as the operating system running within the virtual machine as the guest OS.

Within a virtual machine, AIC100 still requires the assignment of 32 message signal interrupts (MSI) to operate, which requires the virtual machine to emulate an IOMMU. During the creation of a virtual machine, the virtual machine must be configured to emulate a system that can emulate an IOMMU.
 
One such system is the “q35” system. If using “virt-install”, a q35 based virtual machine can be created by adding “—machine=q35” to the virt-install command.

The remainder of this section assumes that the virtual machine is created with virt-install and the –machine=q35 option. Other systems may require different configurations than what is described to obtain the same end effect.

After the virtual machine is created, it must be configured. This can be done with the “virsh edit” command while the virtual machine is not running. See the virsh man page for more details on this command: https://linux.die.net/man/1/virsh.
The virtual machine configuration must have the emulated IOMMU presented to the guest OS in the virtual machine, and the virtual machine must have the AIC100 device passed in.

First, to present the IOMMU to the guest OS, add the following elements to the configuration XML:
 1. Configure the ioapic driver in split mode for interrupt remapping. This is done by adding “<ioapic driver='qemu'/>” under the “features” section of the XML.
 2. Configure the emulated IOMMU to be present with the interrupt remapping functionality enabled. This is done by adding the following snippet under the “features” section of the XML:
 ```
 <iommu model='intel'>
 <driver caching_mode='on' intremap='on'/>
 </iommu>
 ```
Second, configure what device to pass through to the virtual machine guest OS.
 1. Obtain the PCIe SBDF address of the AIC100 device using “lspci” in the host.
 2. Add the device address obtained from Step 1 to the “devices” section of the XML, as follows, but change the address tag values to match that of your specific system.
 ```
 <hostdev mode='subsystem' type='pci' managed='yes'>
 <source>
 <address domain='0x0' bus='0x0' slot='0x19' function='0x0'/>
 </source>
 </hostdev>
 ```
After making these edits, save the configuration. The next time the virtual machine is booted, you will observe the AIC100 device in lspci output in the virtual machine. It will have a different PCIE SBDF address than the host. Install the AIC100 Platform SDK and Apps SDK in the same manner as if you were operating a native system.
