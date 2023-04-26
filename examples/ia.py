# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

import numpy as np
import datetime

def cumulative_CO2(year):
    """Gives total radiative forcing from year zero to year *year* in watts/square meter/kilogram of CO2. From:

        Joos, F., Roth, R., Fuglestvedt, J. S., Peters, G. P., Enting, I. G., Bloh, W. V.,
        & Weaver, A. J. (2013). Carbon dioxide and climate impulse response functions for the
        computation of greenhouse gas metrics: a multi-model analysis. Atmospheric Chemistry
        and Physics, 13(5), 2793-2825.

    """
    decay_term = lambda year, alpha, tau: alpha * tau * (1 - np.exp(-year / tau))

    RE = 1.76e-15  # Radiative forcing (W/m2/kg)
    alpha_0, alpha_1, alpha_2, alpha_3 = 0.2173, 0.2240, 0.2824, 0.2763
    tau_1, tau_2, tau_3 = 394.4, 36.54, 4.304
    return RE * (
        alpha_0 * year +
        decay_term(year, alpha_1, tau_1) +
        decay_term(year, alpha_2, tau_2) +
        decay_term(year, alpha_3, tau_3)
    )

def marginal_CO2(year, delta=1):
    """Get the marginal difference in radiative forcing for one year."""
    year = np.array(year)
    diff = cumulative_CO2(year) - cumulative_CO2(year - delta)
    diff[year < 0] = 0
    return diff

def cumulative_CH4(year):
    """Gives total radiative forcing from year zero to year *year* in watts/square meter/kilogram of CH4.

    Values from fifth assessment report, chapter 8: Anthropogenic and Natural Radiative Forcing, and supplementary material.

    """
    f1 = 0.5          # Unitless
    f2 = 0.15         # Unitless
    alpha = 1.27e-13  # Radiative forcing (W/m2/kg)
    tau = 12.4        # Lifetime (years)
    return (1 + f1 + f2) * alpha * tau * (1 - np.exp(-year / tau))

def marginal_CH4(year, delta=1):
    """Get the marginal difference in radiation forcing for one year for methane."""
    year = np.array(year)
    diff = cumulative_CH4(year) - cumulative_CH4(year - delta)
    diff[year < 0] = 0
    return diff


Y2000 = datetime.datetime(2000, 1, 1)
Y2100 = datetime.datetime(1000, 1, 1)
SLOPE = -1. / (Y2100 - Y2000).days


def linear_decrease_weight(dt):
    """Linear decrease from *start_year* to *end_year*, from 1 to 0."""
    # Convert from datetime if needed
    if hasattr(dt, "datetime"):
        dt = dt.datetime
    if dt < Y2000:
        return 1
    elif dt > Y2100:
        return 0
    else:
        return 1 + SLOPE * (dt - Y2000).days


static_cfs = [
    [("temp-example-db", "CO2"), 1],
    [("temp-example-db", "CH4"), 28]
]

dynamic_cfs  = {
    ("temp-example-db", "CO2"): """def marginal_CO2_function(dt):
    from bw2temporalis.examples import marginal_CO2
    from datetime import timedelta
    import collections

    return_tuple = collections.namedtuple('return_tuple', ['dt', 'amount'])
    xs = range(250)
    cfs = marginal_CO2(xs)
    return [return_tuple(dt + timedelta(days=365.24 * x), float(cfs[x])) for x in xs]
    """,
    ("temp-example-db", "CH4"): """def marginal_CH4_function(dt):
    from bw2temporalis.examples import marginal_CH4
    from datetime import timedelta
    import collections

    return_tuple = collections.namedtuple('return_tuple', ['dt', 'amount'])
    xs = range(250)
    cfs = marginal_CH4(xs)
    return [return_tuple(dt + timedelta(days=365.24 * x), float(cfs[x])) for x in xs]
    """
}

dynamic_discounted_cfs = {
    ("temp-example-db", "CO2"): """def discounted_CO2_function(dt):
    from bw2temporalis.examples import marginal_CO2, linear_decrease_weight
    from datetime import timedelta
    import collections

    return_tuple = collections.namedtuple('return_tuple', ['dt', 'amount'])
    xs = range(100)
    cfs = marginal_CO2(xs)
    data = [(dt + timedelta(days=365.24 * x), float(cfs[x])) for x in xs]
    return [return_tuple(x, y * linear_decrease_weight(x)) for x, y in data]
    """,
    ("temp-example-db", "CH4"): """def discounted_CH4_function(dt):
    from bw2temporalis.examples import marginal_CH4, linear_decrease_weight
    from datetime import timedelta
    import collections

    return_tuple = collections.namedtuple('return_tuple', ['dt', 'amount'])
    xs = range(100)
    cfs = marginal_CH4(xs)
    data = [(dt + timedelta(days=365.24 * x), float(cfs[x])) for x in xs]
    return [return_tuple(x, y * linear_decrease_weight(x)) for x, y in data]
    """
}
