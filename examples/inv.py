# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *
from bw2temporalis import TemporalDistribution
import numpy as np


db_data = {
    ('temp-example-db', "CO2"): {
        "type": "emission"
    },
    ('temp-example-db', "CH4"): {
        "type": "emission"
    },
    ('temp-example-db', 'Functional Unit'): {
        'exchanges': [
            {
                'amount': 5,
                'input': ('temp-example-db', 'EOL'),
                'temporal_distribution': TemporalDistribution(np.array([ 0,  1,  2,  3,  4],dtype='timedelta64[Y]') ,np.array([1.0, 1.0, 1.0, 1.0, 1.0])),

                'type': 'technosphere'
            },
        ],
        'name': 'Functional Unit',
        'type': 'process'
    },
    ('temp-example-db', 'EOL'): {
        'exchanges': [
            {
                'amount': 0.8,
                'input': ('temp-example-db', 'Waste'),
                'type': 'technosphere'
            },
            {
                'amount': 0.2,
                'input': ('temp-example-db', 'Landfill'),
                'type': 'technosphere'
            },
            {
                'amount': 1,
                'input': ('temp-example-db', 'Use'),
                'type': 'technosphere'
            },
        ],
        'name': 'EOL',
        'type': 'process'
    },
    ('temp-example-db', 'Use'): {
        'exchanges': [
            {
                'amount': 1,
                'input': ('temp-example-db', 'Production'),
                'temporal_distribution': TemporalDistribution(np.array([4],dtype='timedelta64[M]') ,np.array([1.0])),
                'type': 'technosphere'
            },
        ],
        'name': 'Use',
        'type': 'process'
    },
    ('temp-example-db', 'Production'): {
        'exchanges': [
            {
                'amount': 1,
                'input': ('temp-example-db', 'Transport'),
                'temporal_distribution': TemporalDistribution(np.array([200],dtype='timedelta64[D]') ,np.array([1.0])),
                'type': 'technosphere'
            },
        ],
        'name': 'Production',
        'type': 'process'
    },
    ('temp-example-db', 'Transport'): {
        'exchanges': [
            {
                'amount': 1,
                'input': ('temp-example-db', 'Sawmill'),
                'type': 'technosphere'
            },
            {
                'amount': 0.1,
                'input': ('temp-example-db', 'CO2'),
                'type': 'biosphere'
            },
        ],
        'name': 'Production',
        'type': 'process'
    },
    ('temp-example-db', 'Sawmill'): {
        'exchanges': [
            {
                'amount': 1.2,
                'input': ('temp-example-db', 'Forest'),
                'temporal_distribution':  TemporalDistribution(np.array([14],dtype='timedelta64[M]') ,np.array([1.2])),
                'type': 'technosphere'
            },
            {
                'amount': 0.1,
                'input': ('temp-example-db', 'CO2'),
                'type': 'biosphere'
            },
        ],
        'name': 'Sawmill',
        'type': 'process'
    },
    ('temp-example-db', 'Forest'): {
        'exchanges': [
            {
                'amount': -.2 * 6,
                'input': ('temp-example-db', 'CO2'),
                'temporal_distribution': TemporalDistribution(np.array([-4,-3,0,1,2,5],dtype='timedelta64[Y]') ,np.array([-.2]*6)),
                'type': 'biosphere'
            },
            {
                'amount': 1.5,
                'input': ('temp-example-db', 'Thinning'),
                'temporal_distribution': TemporalDistribution(np.array([-3,0,1],dtype='timedelta64[Y]') ,np.array([.5]*3)),
                'type': 'technosphere'
            },
        ],
        'name': 'Forest',
        'type': 'process'
    },
    ('temp-example-db', 'Thinning'): {
        'exchanges': [
            {
                'amount': 1,
                'input': ('temp-example-db', 'Thinning'),
                'type': 'production'
            },
            {
                'amount': 1,
                'input': ('temp-example-db', 'Avoided impact - thinnings'),
                'type': 'production'
            },
        ],
        'name': 'Thinning',
        'type': 'process'
    },
    ('temp-example-db', 'Landfill'): {
        'exchanges': [
            {
                'amount': 0.1,
                'input': ('temp-example-db', 'CH4'),
                'temporal_distribution': TemporalDistribution(np.array([10,20,40,60],dtype='timedelta64[M]') ,np.array([0.025]*4)),
                'type': 'biosphere'
            },
        ],
        'name': 'Landfill',
        'type': 'process'
    },
    ('temp-example-db', 'Waste'): {
        'exchanges': [
            {
                'amount': 1,
                'input': ('temp-example-db', 'Waste'),
                'type': 'production'
            },
            {
                'amount': 1,
                'input': ('temp-example-db', 'Avoided impact - waste'),
                'type': 'production'
            },
        ],
        'name': 'Waste',
        'type': 'process'
    },
    ('temp-example-db', 'Avoided impact - waste'): {
        'exchanges': [
            {
                'amount': -0.6,
                'input': ('temp-example-db', 'CO2'),
                'type': 'biosphere'
            },
            {
                'amount': 1,
                'input': ('temp-example-db', 'Avoided impact - waste'),
                'type': 'production'
            },
        ],
        'name': 'Avoided impact - waste',
        'type': 'process'
    },
    ('temp-example-db', 'Avoided impact - thinnings'): {
        'exchanges': [
            {
                'amount': -0.2,
                'input': ('temp-example-db', 'CO2'),
                'type': 'biosphere'
            },
            {
                'amount': 1,
                'input': ('temp-example-db', 'Avoided impact - thinnings'),
                'type': 'production'
            },
        ],
        'name': 'Avoided impact - thinnings',
        'type': 'process'
    }
}
