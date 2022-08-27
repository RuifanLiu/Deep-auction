from mardan.dep import ORTOOLS_ENABLED, pywrapcp, routing_enums_pb2
from mardan.dep import tqdm

from argparse import ArgumentParser
import os.path
from multiprocessing import Pool
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from marpdan.problems import SDVRPTW_Dataset, SDVRPTW_Environment
from marpdan.utils import eval_apriori_routes


DEPOT = 0
LOCX, LOCY, DEM, RDY, DUE, DUR, APR = range(7)
LOC = slice(0, 2)
TIMES = slice(3, 7)


def _solve_cp(nodes, dist, veh_count, veh_capa, veh_speed, pending_cost, late_cost, partial_routes=None, t=0):
    horizon = int(nodes[DEPOT, DUE])
    nodes_count = nodes.size(0)

    vis = nodes[:, APR] <= t
    vis_idx = torch.arange(nodes_count)[vis].tolist()
    vis_count = int(vis.sum())

    manager = pywrapcp.RoutingIndexManager(vis_count, veh_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_i, to_i):
        from_n = manager.IndexToNode(from_i)
        to_n = manager.IndexToNode(to_i)
        return int(dist[vis_idx[from_n], vis_idx[to_n]])
    dist_cb_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    def dem_cb(from_i):
        from_n = manager.IndexToNode(from_i)
        return int(nodes[vis_idx[from_n], DEM])
    dem_cb_idx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimension(dem_cb_idx, 0, veh_capa, True, "Capacity")

    def t_cb(from_i, to_i):
        from_n = manager.IndexToNode(from_i)
        to_n = manager.IndexToNode(to_i)
        return int(nodes[vis_idx[from_n], DUR] + dist[vis_idx[from_n], vis_idx[to_n]] / veh_speed)
    t_cb_idx = routing.RegisterTransitCallback(t_cb)
    routing.AddDimension(t_cb_idx, horizon, 2 * horizon, False, "Time")

    t_dim = routing.GetDimensionOrDie("Time")
    for from_n, vis_i in enumerate(vis_idx[1:]):
        from_i = manager.NodeToIndex(from_n)
        t_dim.CumulVar(from_i).SetMin(int(nodes[vis_i, RDY]))
        t_dim.SetCumulVarSoftUpperBound(from_i, int(nodes[vis_i, DUE]), late_cost)
    for veh_i in range(veh_count):
        end_i = routing.End(veh_i)
        t_dim.SetCumulVarSoftUpperBound(end_i, horizon, late_cost)
        routing.AddVariableMinimizedByFinalizer(t_dim.CumulVar(end_i))

    for to_n in range(1, vis_count):
        routing.AddDisjunction([manager.NodeToIndex(to_n)], pending_cost, 1)

    if partial_routes is not None:
        routing.CloseModel()
        rev_idx = {vis_i:to_n for to_n, vis_i in enumerate(vis_idx)}
        locks = [[rev_idx[vis_i] for vis_i in route] for route in partial_routes]
        routing.ApplyLocksToAllVehicles(locks, False)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(params)

    routes = [[] for _ in range(veh_count)]
    traj = []
    for veh_i in range(veh_count):
        t = 0
        to_i = routing.Start(veh_i)
        while not routing.IsEnd(to_i):
            to_i = solution.Value(routing.NextVar(to_i))
            to_n = manager.IndexToNode(to_i)
            routes[veh_i].append(vis_idx[to_n])
            traj.append((t, veh_i, vis_idx[to_n]))
            t = solution.Value(t_dim.CumulVar(to_i)) + int(nodes[vis_idx[to_n], DUR])
    traj.sort()
    return routes, traj


def _solve_loop(nodes, veh_count, veh_capa, veh_speed, pending_cost, late_cost):
    hidden = nodes[:, APR] > 0
    aprs = nodes[hidden, APR].tolist()
    aprs.sort(reverse=True)

    dist = (nodes[:, None, LOC] - nodes[None, :, LOC]).pow(2).sum(axis=2).pow(0.5).ceil()

    partial = None
    t = 0

    while aprs:
        routes, traj = _solve_cp(nodes, dist, veh_count, veh_capa, veh_speed, pending_cost, late_cost,
                                 partial, t)
        partial = [[] for _ in range(veh_count)]
        for t, veh_i, to_n in traj:
            if t < aprs[-1]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while aprs and t >= aprs[-1]:
                aprs.pop()
            break
        else:
            t = aprs.pop()
    routes, traj = _solve_cp(nodes, dist, veh_count, veh_capa, veh_speed, pending_cost, late_cost, partial, t)
    return routes


def ort_solve_dyna(data, pending_cost=200, late_cost=1, no_mp=False):
    if no_mp:
        routes = []
        for nodes in tqdm(data.nodes):
            routes.append(_solve_loop(nodes, data.veh_count, data.veh_capa, data.veh_speed,
                                      pending_cost, late_cost))
        return routes
    else:
        with Pool() as p:
            with tqdm(desc = "Calling ORTools", total = data.batch_size) as pbar:
                results = [p.apply_async(_solve_loop, (nodes, data.veh_count, data.veh_capa, data.veh_speed,
                                                       pending_cost, late_cost),
                                         callback=lambda _:pbar.update()) for nodes in data.nodes]
                routes = [res.get(timeout=240) for res in results]
    return routes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--normalized-data", "-n", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pending-cost", type=int, default=200)
    parser.add_argument("--late-cost", type=int, default=1)
    parser.add_argument("--no-mp", action="store_true")
    return parser.parse_args()


def main(args):
    data = torch.load(args.data_path)
    assert isinstance(data, SDVRPTW_Dataset)

    if args.normalized_data:
        unnorm = data.nodes.clone()
        unnorm[:, :, LOC] *= 100
        unnorm[:, :, DEM] *= 200
        unnorm[:, :, TIMES] *= 480
        unnorm_data = SDVRPTW_Dataset(data.veh_count, 200, 1, unnorm, data.cust_mask)
    else:
        nodes = data.nodes.clone()
        unnorm_data = SDVRPTW_Dataset(data.veh_count, data.veh_capa, data.veh_speed, nodes, data.cust_mask)
        data.normalize()

    routes = ort_solve_dyna(unnorm_data, args.pending_cost, args.late_cost, args.no_mp)
    torch.save(routes, "DUMP_routes_dyn.pyth")
#    routes = torch.load("DUMP_routes_dyn.pyth")

    loader = DataLoader(data, batch_size=50)
    with tqdm(loader, "Evaluating") as progress:
        costs = []
        qos = []
        for b, batch in enumerate(progress):
            env = SDVRPTW_Environment(data, batch, pending_cost=0, late_p=0)
            costs.append(eval_apriori_routes(env, routes[50*b:50*(b+1)], 1))
            pending = (env.served ^ True).float().sum(-1) - 1
            qos.append(1 - pending / (env.nodes_count - 1))
        costs = torch.cat(costs, 0)
        qos = torch.cat(qos, 0)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.data_path).replace("data", "results")

    dods = (data.nodes[:,:,6] > 0).sum(1).float() / (data.nodes.size(1)-1)
    for k,subset in (("leq40", dods <= 0.4),
            ("less60", (0.4 < dods) & (dods < 0.6)),
            ("geq60", 0.6 <= dods)):
        print("{}: {:5.2f} +- {:5.2f} (QoS: {:.2%})".format(k, costs[subset].mean(), costs[subset].std(),
                                                            qos[subset].mean()))
        torch.save({"costs":costs[subset], "qos":qos[subset], "routes":[routes[b] for b in subset]},
                os.path.join(args.output_dir, "ort_{}.pyth".format(k)))


if __name__ == "__main__":
    main(parse_args())

