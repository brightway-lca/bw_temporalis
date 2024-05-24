# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

from brightway2 import Database, Method, databases, methods
from eight import *

from .. import DynamicIAMethod, dynamic_methods
from .ia import (
    cumulative_CH4,
    cumulative_CO2,
    dynamic_cfs,
    dynamic_discounted_cfs,
    linear_decrease_weight,
    marginal_CH4,
    marginal_CO2,
    static_cfs,
)
from .inv import db_data


def import_example_data():
    db = Database("temp-example-db")
    if db.name not in databases:
        db.register()
    db.write(db_data)
    db.process()

    method = Method(("static GWP",))
    if method.name not in methods:
        method.register()
    method.write(static_cfs)
    method.process()

    dynamic_method = DynamicIAMethod("static GWP")
    if dynamic_method.name not in dynamic_methods:
        dynamic_method.register()
    dynamic_method.write({x[0]: x[1] for x in static_cfs})
    dynamic_method.to_worst_case_method(("static GWP", "worst case"))

    dynamic_method = DynamicIAMethod("dynamic GWP")
    if dynamic_method.name not in dynamic_methods:
        dynamic_method.register()
    dynamic_method.write(dynamic_cfs)
    dynamic_method.to_worst_case_method(("dynamic GWP", "worst case"))

    dynamic_method = DynamicIAMethod("discounted dynamic GWP")
    if dynamic_method.name not in dynamic_methods:
        dynamic_method.register()
    dynamic_method.write(dynamic_discounted_cfs)
    dynamic_method.to_worst_case_method(("discounted dynamic GWP", "worst case"))
