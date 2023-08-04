ESXi is the virtualization hypervisor from VMWare. Refer to ESXi documentation for instructions on how to create a virtual machine (VM).

Before powering-on the VM and installing the guest OS, a few setting changes are required to assign one or more AI 100 cards to the VM. To activate passthrough, follow these steps.
  1. Go to “Host” -> ‘Manage’ tab on the left bar. Then click on the ‘Hardware’ tab.
  2. Search for “Qualcomm”. All the installed AI 100 cards should show up.
  3. Check the cards and click “Toggle Passthrough”. Each card should then list “Active” under the Passthrough attribute.
  4. Create a VM per instructions in the ESXi documentation. Under "virtual hardware", "add other device", and select "pci device". This will add a new entry for the PCI device. Verify that the correct AI 100 card is selected here. Repeat this process for every AI 100 card that should be assigned to the VM.
  5. Setup is now complete and the VM can be powered ON. It should automatically boot the guest OS ISO and start the installer. A preview of the console is shown in the virtual machine tab when the concerned VM is selected. The preview can be clicked to be expanded and used as an interface for the VM. Walk through the OS installer like any other system.
