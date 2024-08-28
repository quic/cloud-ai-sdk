#!/usr/bin/env python3
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pyudev
import re
import json
import os
import stat
import sys
import subprocess

"""
This program disables ACS on pcie branches among AIC100 cards
Usage:
  python3 <QAicChangeAcs.py>
    Displays a hierarchical view of pci bridges and the AIC100 device BDFs
    Steps:
      Identify for which group of devices the ACS needs to be disabled
      Find the nearst ancestor pci bridge of these devices
      Note the bdf info of that common pci bridge
  python3 QAicChangeAcs.py SSSS:BB:DD.F
    where
        SSSS = 4 digits segment number
        BB = 2 digits bus number
        DD = 2 digits device number
        F = 1 digit function number
      of the nearst common ancestor pci bridge under which ACS needs to be disabled
  python3 QAicChangeAcs.py all
    Disables all possible ACS systemwide
  python3 QAicChangeAcs.py SSSS:BB:DD.F enable  # Non-default case. Multi-Device inference in P2P will not work
    Enables Acs.

Example:
$ python3 QAicChangeAcs.py
Found the following AIC100 devices:
Root
    0000:30:01.1
        0000:31:00.0
            0000:32:03.0
                0000:36:00.0 [Qualcomm AIC100]
            0000:32:02.0
                0000:35:00.0 [Qualcomm AIC100]
            0000:32:00.0
                0000:38:00.0 [Qualcomm AIC100]
            0000:32:01.0
                0000:39:00.0 [Qualcomm AIC100]
    0000:00:01.1
        0000:21:00.0
            0000:22:00.0
                0000:23:00.0 [Qualcomm AIC100]
            0000:22:02.0
                0000:25:00.0 [Qualcomm AIC100]
            0000:22:01.0
                0000:27:00.0 [Qualcomm AIC100]
            0000:22:03.0
                0000:28:00.0 [Qualcomm AIC100]
$ python3 QAicChangeAcs.py 0000:31:00.0 # To disable the first set of AIC100 (:36, :35, :38, :39) because :31 is the nearest bridge that encompasses all
Disabling ACS on:
0000:32:03.0
0000:32:02.0
0000:32:00.0
0000:32:01.0
"""

# Contains all devices. Includes parent and children. Dict {self: <BDF>, children: list{}}
allDevices = list()

# List of the resultant pci bdfs for disbling acs
content = list()

enableAcs = 0 # Disable by default

def dispCore(allBdfs, bdf, indent):
  foundDev = {}
  for elem in allBdfs:
    if elem["self"] == bdf:
      foundDev = elem
      break
  if not foundDev == {}:
    if len(foundDev["children"]):
      print((" " * indent) + foundDev["self"])
    else:
      print((" " * indent) + foundDev["self"] + " [Qualcomm AIC100]")
    for child in foundDev["children"]:
      dispCore(allBdfs, child, indent + 5)

def disp(allBdfs):
  rootDev = dict(self = "Root", children = set())
  for elem in allBdfs:
    if elem["parent"] == "Root":
      rootDev["children"].add(elem["self"])
  allBdfs.append(rootDev)
  dispCore(allBdfs, "Root", 0)

def changeAcs():
  global content
  global enableAcs
  for bdf in content:
    stsCmd = ["sudo", "setpci", "-s", bdf, "ecap_acs+6.w=" + str(enableAcs)]
    # stsCmd = ["sudo", "setpci", "-s", bdf, "ecap_acs+6.w"]
    try:
      subprocess.check_output(stsCmd, stderr = subprocess.DEVNULL)
    except subprocess.CalledProcessError as errMsg:
      if not str(errMsg).find("there are no capabilities with that id"):
        print("Error: unexpected error in checking extended capability register.")
        exit()
    print(bdf)

  if not enableAcs:
    # Also call qaic-util -a to directly configure the Ultra AI 100 onboard PCIe switches.
    # This is needed since some upstream PCIe switch firmware versions might block ACS commands
    # from propagating to the Ultra AI 100 cards
    stsCmd = ["sudo", "/opt/qti-aic/tools/qaic-util", "-a"]
    try:
      subprocess.check_output(stsCmd, stderr = subprocess.DEVNULL)
    except subprocess.CalledProcessError as errMsg:
      print("Error: unexpected error disabling ACS on onboard PCIe switches.")
      print(str(errMsg))
      exit()

def getBdfFromDev(device):
  bdfStr = device.get("PCI_SLOT_NAME") 
  return bdfStr

def populateAncestors(device):
  global allDevices
  if device == None:
    return
  parent = device.find_parent("pci")
  while not parent == None:
    allDevices.append({
      "device": device,
      "parent": parent
    })
    device = parent
    parent = device.find_parent("pci")
  if parent == None:
    allDevices.append({
      "device": device,
      "parent": None
    })

firstOneSkipped = False
def findTree(bdf):
  global firstOneSkipped
  global allDevices
  global content
  foundDev = {}
  for elem in allDevices:
    if elem["self"] == bdf:
      foundDev = elem
      break
  if foundDev == {}:
    print("Error: Could not find device: " + bdf)
    return
  if firstOneSkipped:
    if len(foundDev["children"]):
      content.append(bdf)
  else:
    firstOneSkipped = True
  for child in foundDev["children"]:
    findTree(child)

"""
Uses pyudev to query all devices, filter by accel, find pci parent bridge
Then converts to a list of bdfs as string including children info
"""
def main():
  global allDevices
  global enableAcs
  context = pyudev.Context()
  devices = context.list_devices()
  devices.match_subsystem("accel")
  for d in devices:
    if d.driver == "qaic":
      populateAncestors(d)
    else:
      d = d.find_parent("pci")
      if d.driver == "qaic":
        populateAncestors(d)
  
  # Transform from pyudev data to simple string of bdfs
  allDevices = [{"self": getBdfFromDev(elem["device"]), "children": set(), "parent": "Root"} if elem["parent"] == None else {"self": getBdfFromDev(elem["device"]), "children": set(), "parent": getBdfFromDev(elem["parent"])} for elem in allDevices]
  """ Schema:
  [{
    self: SSSS:BB:DD.F,
    children: set(SSSS:BB:DD.F, ),
    parent: SSSS:BB:DD.F
  }, ...]
  """

  # Transform from parent relation to children relation
  for child in allDevices:
    for parent in allDevices:
      if child["parent"] == parent["self"]:
        parent["children"].add(child["self"]) 
        break
  
  if len(sys.argv) == 1:
    print("Found the following AIC100 devices:")
    disp(allDevices)
    exit()
  
  inputBdf = sys.argv[1]

  if len(sys.argv) == 3:
    enableAcs = "001d"
  
  # Find the downstream tree starting from the given pci bridge bdf
  if inputBdf.lower() == "all":
    firstBorns = set()
    for elem in allDevices:
      if elem["parent"] == "Root":
        firstBorns.add(elem["self"])
    for elem in firstBorns:
      findTree(elem)
  else:
    findTree(inputBdf)

  if not enableAcs:
    print("Disabling ACS on:")
  else:
    print("Enabling ACS on:")

  changeAcs()

if __name__ == "__main__":
  main()

