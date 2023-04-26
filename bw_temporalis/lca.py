from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from heapq import heappop, heappush

import bw2data as bd
import numpy as np
from bw2calc import LCA
from bw2data.backends import ActivityDataset as AD
from bw2data.backends import ExchangeDataset as ED
from bw_graph_tools import GraphTraversal

from .temporal_distribution import TemporalDistribution as TD
from .timeline import Timeline


class MultipleTechnosphereExchanges(Exception):
    pass


class TemporalisLCA:
    # TBD
    """Calculate a dynamic LCA, where processes, emissions, and CFs can vary throughout time.

    Args:
        * *demand* (dict): The functional unit. Same format as in LCA class.
        * *worst_case_method* (tuple): LCIA method. Same format as in LCA class.
        * *cutoff* (float, default=0.005): Cutoff criteria to stop LCA calculations. Relative score of total, i.e. 0.005 will cutoff if a dataset has a score less than 0.5 percent of the total.
        * *max_calc_number* (int, default=10000): Maximum number of LCA calculations to perform.
        * *loop_cutoff* (int, default=10): Maximum number of times loops encountered will be traversed.
        * *t0* (datetime, default=np.datetime64('now')): `datetime` of the year zero (i.e. the one of the functional unit).
        * *group* (Boolean, default=False): When 'True' groups the impact upstream for each of the processes based on the values of `grouping_field`
        * *grouping_field* (string, default='tempo_group': The bw2 field to look for when grouping impacts upstream. When ``group`==True and a process has `grouping_field==whatever` the impacts are grouped upstream with name ``whatever` untill another  process with `grouping_field==another name` is found. If `grouping_field==True` it simply uses the name of the process
        * *log* (int, default=False): If True to make log file
        * *lca_object* (LCA object,default=None): do dynamic LCA for the object passed (must have "characterized_inventory" i.e. LCA_object.lcia() has been called)
    """

    def __init__(
        self,
        lca_object: LCA,
        starting_datetime: datetime | str = "now",
        graph_traversal_config: dict | None = None,
        cutoff: float | None = 5e-3,
        biosphere_cutoff: float | None = 1e-5,
        max_calc: int | None = 1e5,
        static_activity_indices: set[int] | None = set(),
        skip_coproducts: bool | None = False,
        functional_unit_unique_id: int | None = -1,
    ):
        self.lca_object = lca_object
        self.unique_id = functional_unit_unique_id
        self.t0 = TD(
            np.array([np.datetime64(starting_datetime)]).astype("datetime64[D]"),
            np.array([1]),
        )
        for db in bd.databases:
            if bd.databases[db].get("static"):
                static_activity_indices.add(
                    {
                        obj[0]
                        for obj in AD.select(AD.id).where(AD.database == db).tuples()
                    }
                )

        print("Starting graph traversal")
        gt = GraphTraversal.calculate(
            lca_object=lca_object,
            static_activity_indices=static_activity_indices,
            max_calc=max_calc,
            cutoff=cutoff,
            biosphere_cutoff=biosphere_cutoff,
            separate_biosphere_flows=True,
            skip_coproducts=skip_coproducts,
            functional_unit_unique_id=functional_unit_unique_id,
        )
        print("Calculation count:", gt["calculation_count"])
        self.nodes = gt["nodes"]

        self.edges = gt["edges"]
        self.edge_mapping = defaultdict(list)
        for edge in self.edges:
            self.edge_mapping[edge.consumer_id].append(edge)

        self.flows = gt["flows"]
        self.flow_mapping = defaultdict(list)
        for flow in self.flows:
            self.flow_mapping[flow.activity_unique_id].append(flow)

    def build_timeline(self) -> Timeline:
        heap = []
        timeline = Timeline()

        for edge in self.edge_mapping[self.unique_id]:
            node = self.nodes[edge.producer_id]
            heappush(
                heap,
                (
                    1 / node.cumulative_score,
                    self.t0 * edge.amount,
                    node,
                ),
            )

        while heap:
            _, td, node = heappop(heap)

            for flow in self.flow_mapping.get(node.unique_id, []):
                for exchange in self.get_biosphere_exchanges(
                    flow.flow_datapackage_id, node.activity_datapackage_id
                ):
                    value = (
                        exchange.data.get("temporal distribution")
                        or exchange.data["amount"]
                    )
                    timeline.add_flow_temporal_distribution(
                        td=td * value,
                        flow=flow.flow_datapackage_id,
                        activity=node.activity_datapackage_id,
                    )

            for edge in self.edge_mapping[node.unique_id]:
                exchange = self.get_technosphere_exchange(
                    input_id=self.nodes[edge.producer_id].activity_datapackage_id,
                    output_id=node.activity_datapackage_id,
                )
                value = (
                    exchange.data.get("temporal distribution")
                    or exchange.data["amount"]
                ) / node.reference_product_production_amount
                producer = self.nodes[edge.producer_id]
                heappush(
                    heap,
                    (
                        1 / node.cumulative_score,
                        td * value,
                        producer,
                    ),
                )
        return timeline

    def _exchange_iterator(self, input_id: int, output_id: int) -> list[ED]:
        inp = AD.get(AD.id == input_id)
        outp = AD.get(AD.id == output_id)
        return list(
            ED.select().where(
                ED.input_code == inp.code,
                ED.input_database == inp.database,
                ED.output_code == outp.code,
                ED.output_database == outp.database,
            )
        )

    def get_biosphere_exchanges(self, flow_id: int, activity_id: int) -> Iterable[ED]:
        exchanges = self._exchange_iterator(flow_id, activity_id)
        if len(exchanges) > 1:
            total = sum(exc.data["amount"] for exc in exchanges)
            for exc in exchanges:
                exc.data["amount"] /= total
                if "temporal distribution" in exc.data:
                    exc.data["temporal distribution"] /= total
                yield exc
        else:
            return exchanges[0]

    def get_technosphere_exchange(self, input_id: int, output_id: int) -> ED:
        exchanges = self._exchange_iterator(input_id, output_id)
        if len(exchanges) > 1:
            _ = lambda x: "{}|{}|{}".format(x.database, x.name, x.code)
            raise MultipleTechnosphereExchanges(
                "Found {} exchanges for link between {} and {}".format(
                    len(exchanges), _(inp), _(outp)
                )
            )
        return exchanges[0]
