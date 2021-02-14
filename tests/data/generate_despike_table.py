#! /usr/bin/env python

import pandas as pd
import numpy as np

#spur_width=120
spur_width=4
outlier_ch=np.array([np.arange(6000,8000)])

startscan=21280
endscan=21280
maxsubscan=10
instmode='OTFSWA'

npix=7

firstflg=1
rows = []
for scan in range(startscan,endscan+1):
    for subscan in range(0,maxsubscan+2,2):
        for ipix in range(npix):
            telescope = "LFAV_PX{:02d}_S".format(ipix)
            for instmode in ['OTFSWA','CAL']:
              rows.append({"scannum" : scan, 
                           "subscan" : subscan,
                           "backend" : telescope,
                           "instmode" : instmode,
                           'outliers_width_{0}'.format(spur_width) : outlier_ch
                           })

df = pd.DataFrame(rows)


pd.to_pickle(df,'despike_lookup_table.pkl')
