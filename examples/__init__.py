# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *


from brightway2 import Database, databases, Method, methods
from .. import dynamic_methods, DynamicIAMethod
from .inv import db_data
from .ia import static_cfs, dynamic_cfs, dynamic_discounted_cfs
from .ia import cumulative_CO2, marginal_CO2, cumulative_CH4, marginal_CH4, linear_decrease_weight


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
