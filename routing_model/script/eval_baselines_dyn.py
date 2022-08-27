from marpdan.problems import SDVRPTW_Environment
from marpdan.dep import tqdm
from marpdan.externals._ort import _solve_cp

import torch
from torch.utils.data import DataLoader
from multiprocessing import Pool

def best_insert(nodes, states, routes, insert_mask, threshold=2):
    for inst_nodes, inst_states, inst_routes, inst_mask in zip(nodes, states, routes, insert_mask):
        inst_dist = []
        inst_time = []
        inst_costs = []
        for i, (veh, route) in enumerate(zip(inst_states, inst_routes)):
            if not route:
                route = [0]
            route_nodes = inst_nodes[route]
            dist = torch.cat((veh[:2].unsqueeze(0), route_nodes[:, :2]), 0)
            dist = (dist[:-1] - dist[1:]).pow(2).sum(1).sqrt().tolist()
            time = [veh[3].item()]
            late = 0
            for k,j in enumerate(route):
                t = max(time[k] + dist[k], route_nodes[k, 3].item())
                late += max(t - route_nodes[k, 4].item(), 0)
                time.append((t + route_nodes[k, 5].item()))
            inst_dist.append(dist)
            inst_time.append(time)
            inst_costs.append(sum(dist) + late)

        for ins_j in inst_mask.nonzero():
            ins_node = inst_nodes[ins_j]
            best = float('inf')
            for i, (veh, route, dist, time, c) in enumerate(
                    zip(inst_states, inst_routes, inst_dist, inst_time, inst_costs)):
                if not route:
                    route = [0]
                route_nodes = inst_nodes[route]
                pos = torch.cat((veh[:2].unsqueeze(0), route_nodes[:, :2]), 0)

                for ins_at, _ in enumerate(route):
                    detour = (pos[ins_at:ins_at+2] - ins_node[:, :2]).pow(2).sum(1).sqrt()
                    t = max(time[ins_at] + detour[0], ins_node[0, 3])
                    late = max(t - ins_node[0, 4], 0)
                    t += ins_node[0, 5]
                    t = max(t + detour[1], route_nodes[ins_at, 3])
                    late += max(t - route_nodes[ins_at, 4], 0)
                    t += route_nodes[ins_at, 5]
                    for k,j in enumerate(route[ins_at+1:], start=ins_at+1):
                        t = max(t + dist[k], route_nodes[k, 3])
                        late += max(t - route_nodes[k, 4], 0)
                        t += route_nodes[k, 5]
                    delta_cost = sum(dist) + detour.sum() - dist[ins_at] + late - c
                    if delta_cost < best:
                        best = delta_cost
                        best_i = i
                        best_at = ins_at

            if best < threshold:
                if inst_routes[best_i]:
                    inst_routes[best_i].insert(best_at, ins_j.item())
                else:
                    inst_routes[best_i] = [ins_j.item(), 0]
                route_nodes = inst_nodes[inst_routes[best_i]]
                inst_dist[best_i][best_at] = (route_nodes[best_at+1, :2] - ins_node[0, :2]).pow(2).sum().sqrt()
                inst_dist[best_i].insert(best_at,
                        (route_nodes[best_at-1, :2] - ins_node[0, :2]).pow(2).sum().sqrt() )
                inst_time[best_i].insert(best_at, (max(inst_time[best_i][best_at-1] + inst_dist[best_i][best_at],
                    ins_node[0, 3]) + ins_node[0, 5]).item())
                for k,j in enumerate(inst_routes[best_i][best_at+1:], start=best_at+1):
                    inst_time[best_i][k] = (max(inst_time[best_i][k-1] + inst_dist[best_i][k], route_nodes[k, 3])
                            + route_nodes[k, 5]).item()


def eval_best_insert(env):
    env.reset()
    nodes = env.nodes.clone()
    nodes[:,:,:2] *= 100
    nodes[:,:, 2] *= 200
    nodes[:,:,3:] *= 480
    with Pool() as p:
        results = [p.apply_async(_solve_cp, (n[m^1], env.veh_count, 200, 1, 1))
                for n, m in zip(nodes, env.cust_mask)]
        routes = [res.get() for res in results]
    known_cust = [(m^1).nonzero().squeeze(1).tolist() for m in env.cust_mask]
    routes = [[[known_cust[b][j] for j in route] for route in inst_routes] for b,inst_routes in enumerate(routes)]
    rewards = []
    while not env.done:
        cust_idx = env.cur_veh_idx.new_tensor([[routes[b][i].pop(0) if routes[b][i] else 0]
            for b,i in enumerate(env.cur_veh_idx)])
        prv_hidden = env.nodes[:, :, 6] > env.cur_veh[:, :, 3]
        rewards.append( env.step(cust_idx) )
        if env.new_customers:
            insert_mask = prv_hidden - (env.nodes[:, :, 6] > env.cur_veh[:, :, 3])
            best_insert(env.nodes, env.vehicles, routes, insert_mask)
    return -torch.stack(rewards).sum(0).squeeze(-1)


for n in (10, 20, 50):
    data = torch.load("./data/sd_cvrptw_n{}m{}/norm_data.pyth".format(n, n // 5))
    loader = DataLoader(data, batch_size=50)

    costs = []
    qos = []
    for b,batch in enumerate(tqdm(loader)):
        env = SDVRPTW_Environment(data, batch, pending_cost=0)
        costs.append( eval_best_insert(env) )
        pending = (env.served ^ 1).sum(-1) - 1
        qos.append(1 - pending.float() / (env.nodes_count - 1))

    costs = torch.cat(costs, 0)
    qos = torch.cat(qos, 0)

    dods = (data.nodes[:,:,6] > 0).sum(1).float() / (data.nodes.size(1)-1)
    for k,subset in (("leq40", dods <= 0.4),
            ("less60", (0.4 < dods) & (dods < 0.6)),
            ("geq60", 0.6 <= dods)):
        print("{}: {:5.2f} +- {:5.2f}".format(k, costs[subset].mean(), costs[subset].std()))
        torch.save({"costs":costs[subset], "qos":qos[subset]},
                "./results/sd_cvrptw_n{}m{}/best_insert_{}.pyth".format(n, m, k))
