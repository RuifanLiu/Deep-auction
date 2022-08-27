import torch

import os.path
from itertools import repeat, zip_longest

def actions_to_routes(actions, batch_size, veh_count):
    routes = [[[] for i in range(veh_count)] for b in range(batch_size)]
    for veh_idx, cust_idx in actions:
        for b, (i,j) in enumerate(zip(veh_idx, cust_idx)):
            routes[b][i.item()].append(j.item())
    return routes


def routes_to_string(routes):
    return '\n'.join(
        '0 -> ' + ' -> '.join(str(j) for j in route)
        for route in routes
        )


def export_train_test_stats(args, start_ep, train_stats, test_stats):
    fpath = os.path.join(args.output_dir, "loss_gap.csv")
    with open(fpath, 'a') as f:
        f.write( (' '.join("{: >16}" for _ in range(9)) + '\n').format(
            "#EP", "#LOSS", "#PROB", "#VAL", "#BL", "#NORM", "#TEST_MU", "#TEST_STD", "#TEST_GAP"
            ))
        for ep, (tr,te) in enumerate( zip_longest(train_stats, test_stats, fillvalue=float('nan')), start = start_ep):
            f.write( ("{: >16d}" + ' '.join("{: >16.3g}" for _ in range(8)) + '\n').format(
                ep, *tr, *te))


def _pad_with_zeros(src_it):
    yield from src_it
    yield from repeat(0)


def eval_apriori_routes(dyna, routes, rollout_count):
    mean_cost = dyna.nodes.new_zeros(dyna.minibatch_size)
    for c in range(rollout_count):
        dyna.reset()
        routes_it = [[_pad_with_zeros(route) for route in inst_routes] for inst_routes in routes]
        rewards = []
        while not dyna.done:
            cust_idx = dyna.nodes.new_tensor([[next(routes_it[n][i.item()])]
                for n,i in enumerate(dyna.cur_veh_idx)], dtype = torch.int64)
            rewards.append( dyna.step(cust_idx) )
        mean_cost += -torch.stack(rewards).sum(dim = 0).squeeze(-1)
    return mean_cost / rollout_count


def load_old_weights(learner, state_dict):
    learner.load_state_dict(state_dict)
    for layer in learner.cust_encoder.children():
        layer.mha._inv_sqrt_d = layer.mha.key_size_per_head**0.5
    learner.fleet_attention._inv_sqrt_d = learner.fleet_attention.key_size_per_head**0.5
    learner.veh_attention._inv_sqrt_d = learner.veh_attention.key_size_per_head**0.5

