#!/usr/bin/env python
'''
Created on Nov 7, 2014

@author: cmccully
'''

import os
from pyraf import iraf

iraf.cd(os.getcwd())
iraf.gemini()
iraf.gmos()
iraf.onedspec()

bluecut = 3450

iraf.gmos.logfile = "log.txt"
iraf.gmos.mode = 'h'
iraf.set(clobber='yes')

iraf.set(stdimage='imtgmos')

dooverscan = False
is_GS = False
do_qecorr = False
dobias = False
dodq = False

xchip_shifts = [0.0, 0.0, 0.0]
ychip_shifts = [0.0, 0.0, 0.0]
chip_rotations = [0.0, 0.0, 0.0]
chip_gap_size = 0.0
namps = 0