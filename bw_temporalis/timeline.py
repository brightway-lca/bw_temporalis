# from .dynamic_ia_methods import DynamicIAMethod, dynamic_methods
from bw2data import Method, methods, get_node
import collections
import itertools
from bw_temporalis.temporal_distribution import TemporalDistribution
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class FlowTD:
    distribtion: TemporalDistribution
    flow: int
    activity: int


# TBD: Decorator wrapper for building dataframe


class Timeline:
    """Sum and group elements over time.
    Timeline calculations produce a list of [(datetime, amount)] tuples."""

    def __init__(self, data: list[FlowTD] | None = None):
        self.data = data or []
        self.characterized = []
        self.finalized = False

    def sort(self):
        """Sort the raw timeline data. Characterized data is already sorted."""
        self.raw.sort(key=lambda x: x.dt)

    def add_flow_temporal_distribution(self, td: TemporalDistribution, flow: int, activity: int):
        self.data.append(FlowTD(distribution=td.nonzero(), flow=flow, activity=activity))

    def build_dataframe(self):
        times = np.hstack([o.distribution.times for o in self.data])
        data = np.hstack([o.distribution.data for o in self.data])
        flows = np.hstack([o.flow * np.ones(len(o.distribution)) for o in self.data])
        activities = np.hstack([o.activity * np.ones(len(o.distribution)) for o in self.data])

        self.df = pd.DataFrame({
            'times': times,
            'data': data,
            'flows': flows,
            'activities': activities
        })

    def filter(self, flow: int | None = None, activity: int | None = None):
        f = lambda x: x.flow == flow if flow is not None else True
        g = lambda x: x.activity == activity if activity is not None else True
        return Timeline([x for x in self.raw if f(x) and g(x)])

    def total(self, flow: int | None = None, activity: int | None = None):
        """Return cumulative amount of the flow passed for the activity passed"""
        f = lambda x: x.flow == flow if flow is not None else True
        g = lambda x: x.activity == activity if activity is not None else True
        return sum([x.value for x in self.raw if f(x) and g(x)])

    def characterize_static(self, method, cumulative=True, stepped=False):
        """Characterize a Timeline object with a static impact assessment method.

        Args:
            * *method* (tuple): The static impact assessment method.
            * *cumulative* (bool; default=True): when True return cumulative impact over time.
            * *stepped* (bool; default=True):...
        """
        if method not in methods:
            raise ValueError(f"LCIA static method {method} not found")
        cfs = {get_node(x[0]).id: x[1] for x in Method(method).load()}
        grouped = collections.defaultdict(float)
        for point in self.raw:
            grouped[(point.dt, point.flow)] += point.value

        self.characterized = [GroupedPoint(dt=dt, flow=flow, value=value * cfs[flow])
                              for (dt, flow), value in grouped.items()]
        self.characterized.sort(key=lambda x: x.dt)
        return self._summer(self.characterized, cumulative, stepped)


    def characterize_dynamic(self, method, data=None, cumulative=True, stepped=False,bio_st_emis_yr=None,bio_st_decay=None,rot_stand=None):
        """Characterize a Timeline object with a dynamic impact assessment method.
        Return a nested list of year and impact
        Args:
            * *method* (tuple): The dynamic impact assessment method.
            * *data* (Timeline object; default=None): ....
            * *cumulative* (bool; default=True): when True return cumulative impact over time.
            * *stepped* (bool; default=True):...
            * *bio_st_emis_yr* (int; default=None): year when the biogenic carbon from stand is emitted, by default at yr=0.
            * *bio_st_decay* (str; default=None): emission profile of biogenic carbon from stand .
            * *rot_stand* (int; default=None): lenght of rotation of forest stands.

        """
        if method not in dynamic_methods:
            raise ValueError(u"LCIA dynamic method %s not found" % method)
        meth = DynamicIAMethod(method)
        self.method_data = meth.load()
        #update biogenic carbon profile based on emission year and decay profile if passed
        if any(v is not None for v in (bio_st_emis_yr,bio_st_decay,rot_stand)):
            self.method_data[('static_forest', 'C_biogenic')]="""def custom_co2bio_function(datetime):
                from bw2temporalis.dyn_methods.metrics import {0}
                from datetime import timedelta
                import numpy as np
                import collections
                custom_co2bio_rf_td={0}("co2_biogenic", np.array((1.,)), np.array(({1} or 0,),dtype=('timedelta64[Y]')), 'Y', 1000,{3} or 100,'{2}' or 'delta')
                return_tuple = collections.namedtuple('return_tuple', ['dt', 'amount'])
                return [return_tuple(d,v) for d,v in zip((datetime+custom_co2bio_rf_td.times.astype(timedelta)),custom_co2bio_rf_td.values)]""".format(method,bio_st_emis_yr,bio_st_decay,rot_stand)

        method_functions = meth.create_functions(self.method_data)
        self.characterized = []
        self.dp_groups=self._groupby_sum_by_flow(self.raw if data is None else data)

        for obj in self.dp_groups:
            # if obj.flow in method_functions:
            self.characterized.extend([
                grouped_dp(
                    item.dt,
                    obj.flow,
                    item.amount * obj.amount
                ) for item in method_functions[obj.flow](obj.dt)
            ])
            # else:
                # continue
                #GIU: I would skipe this,we save time plus memory, in groupby_sum_by_flow already skips datapoint not in method_data
                #also more consistent in my opinion (the impact is not 0 but is simply not measurable)

                # self.characterized.append(grouped_dp(
                    # obj.dt,
                    # obj.flow,
                    # obj.amount * method_data.get(obj.flow, 0)
                # ))

        self.characterized.sort(key=lambda x: x.dt)

        return self._summer(self.characterized, cumulative, stepped)

    def characterize_static_by_process(self, method, characterize_static_kwargs={}):
        """Characterize a Timeline object with a static impact assessment method separately by process
        Return a dictionary with process name as key and a nested list of year and impact as value
        Args:
            * *method* (tuple): The static impact assessment method.
            * *characterize_static_kwargs* (dictionary; default={}): optional arguments (passed as key=argument name, value= argument value) passed to the called function `characterize_static` (e.g.'cumulative':True). See `characterize_static` for the possible arguments to pass
        """
        res_st_pr={}
        for pr in self.processes():
            #calculate impact
            imp_pr=self.timeline_for_activity(pr).characterize_static(method,**characterize_static_kwargs)
            if not imp_pr[0]:
                continue
            self._add_results(res_st_pr,pr,imp_pr)
        return res_st_pr

    def characterize_dynamic_by_process(self, method, characterize_dynamic_kwargs={}):
        """Characterize a Timeline object with a static impact assessment method separately by process
        Return a dictionary with process name as key and a nested list of year and impact as value
        Args:
            * *method* (tuple): The dynamic impact assessment method.
            * *characterize_dynamic_kwargs* (dictionary; default={}): optional arguments (passed as key=argument name, value= argument value) passed to the called function `characterize_dynamic` (e.g. 'cumulative':True). See `characterize_dynamic` for the possible arguments to pass
        """
        res_dyn_pr={}
        for pr in self.processes():
            #calculate impact
            imp_pr=self.timeline_for_activity(pr).characterize_dynamic(method,**characterize_dynamic_kwargs)
            if not imp_pr[0]:
                continue
            self._add_results(res_dyn_pr,pr,imp_pr)
        return res_dyn_pr

    def characterize_static_by_flow(self, method, characterize_static_kwargs={}):
        """Characterize a Timeline object with a static impact assessment method separately by flow
        Return a dictionary with flow name as key and a nested list of year and impact as value
        Args:
            * *method* (tuple): The static impact assessment method.
            * *characterize_static_kwargs* (dictionary; default={}): optional arguments (passed as key=argument name, value= argument value) passed to the called function `characterize_static` (e.g.'cumulative':True). See `characterize_static` for the possible arguments to pass
        """
        res_st_fl={}
        for flow in self.flows():
            if flow is not None and flow in self.method_data:
                #calculate impact
                imp_flow=self.timeline_for_flow(flow).characterize_static(method,**characterize_static_kwargs)
                if not imp_flow[0]:
                    continue
                self._add_results(res_st_fl,flow,imp_flow)
        return res_st_fl

    def characterize_dynamic_by_flow(self, method, characterize_dynamic_kwargs={}):
        """Characterize a Timeline object with a static impact assessment method separately by flow
        Return a dictionary with flow name as key and a nested list of year and impact as value
        Args:
            * *method* (tuple): The dynamic impact assessment method.
            * *characterize_dynamic_kwargs* (dictionary; default={}): optional arguments (passed as key=argument name, value= argument value) passed to the called function `characterize_dynamic` (e.g. 'cumulative':True). See `characterize_dynamic` for the possible arguments to pass
        """
        res_dyn_fl={}
        for flow in self.flows():
            if flow is not None and flow in self.method_data:
                #calculate impact
                imp_flow=self.timeline_for_flow(flow).characterize_dynamic(method,**characterize_dynamic_kwargs)
                if not imp_flow[0]:
                    continue
                self._add_results(res_dyn_fl,flow,imp_flow)
        return res_dyn_fl

##############
#INTERNAL USE#
##############

    def _add_results(self,res_dict,key,imp):
        """used in charachterize by flow and processes to groupby impact by year for each flow"""
        if get_activity(key)['name'] not in res_dict:
            #just add to dict
            res_dict[get_activity(key)['name']]=imp
        else:
            #groupby and sum impact by year
            imp[0].extend(res_dict[get_activity(key)['name']][0])
            imp[1].extend(res_dict[get_activity(key)['name']][1])
            c = collections.defaultdict(int)
            for yr,im in zip(imp[0],imp[1]):
                c[yr] += im
            res_dict[get_activity(key)['name']]=tuple(list(t) for t in zip(*sorted(zip(c.keys(), c.values())))) #from https://stackoverflow.com/a/9764364/4929813

    #~1.5 times faster than using Counter() and ~3 than using groupby that need sorting before but still not great (e.g. pandas ~2 times faster, check if possible to use numpy_groupies somehow )
    #CHECK WHAT IS THIS APPROACH AND IF APPLICABLE http://stackoverflow.com/a/18066479/4929813
    def _groupby_sum_by_flow(self,iterable):
        """group and sum datapoint by flow, it makes much faster characterization"""
        c = collections.defaultdict(int)
        for dp in iterable:
            c[dp.dt,dp.flow] += dp.amount
        return[grouped_dp(dt_fl[0],dt_fl[1],amount) for dt_fl,amount in c.items()
                        if dt_fl[1] in self.method_data and amount != 0 ] # skip datapoints with flows without dyn_met and 0 bio_flows

    def _summer(self, iterable: list, cumulative: bool, stepped: bool | None=False):
        if cumulative:
            data = self._cumsum_amount_over_time(iterable)
        # ~##extend to 1000 years if lenght is less than 1000 years
            # ~il len(data)<1000:
                # ~leng=1000-len(data)
                # ~list(range(int(max(res[0]))+1,int(min(res[0])+1000)))

                # ~data[0]+=data[0][-1:]*leng
                # ~data[1]+=data[1][-1:]*leng
        else:
            data =  self._sum_amount_over_time(iterable)
        if stepped:
            return self._stepper(data)
        else:
            return self._to_year([x[0] for x in data]), [x[1] for x in data]

    def _to_year(self, lst):
        """convert datetime to fractional years"""
        to_yr = lambda x: x.year + x.month / 12. + x.day / 365.24
        return [to_yr(obj) for obj in lst]

    def _stepper(self, iterable):
        xs, ys = zip(*iterable)
        xs = list(itertools.chain(*zip(xs, xs)))
        ys = [0] + list(itertools.chain(*zip(ys, ys)))[:-1]
        return self._to_year(xs), ys

    def _sum_amount_over_time(self, iterable):
        """groupby date and sum amount"""
        #GIU: Think here, wiith different data structure we could use consolidate maybe
        return sorted([
            (dt, sum([x.amount for x in res]))
            for dt, res in
            itertools.groupby(iterable, key=lambda x: x.dt.date())
            # itertools.groupby(iterable, key=lambda x: x.dt.astype(object).date()) # to work with numpy datetime. leave for now

        ])

    def _cumsum_amount_over_time(self, iterable):
        """"""
        data = self._sum_amount_over_time(iterable)
        values = [float(x) for x in np.cumsum(np.array([x[1] for x in data]))]
        res=list(zip([x[0] for x in data], values))
        #ffill
        # ~extend=list(range(int(max(res[0]))+1,int(min(res[0])+1000)))
        # ~res[0]+=extend
        # ~res[1]+=res[1][-1:]*len(extend)
        return res
