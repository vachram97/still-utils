#!/usr/bin/env python3

import h5py
import numpy

a = h5py.File('images/ue_191206_SFX-r0006-c00.cxi', 'r') 
x = np.array(a['entry_1/data_1/data'])