#!/usr/bin/env python3


from multiprocessing.sharedctypes import Value
from problems import *
from routing_model.baselines import *
from routing_model.externals import *
from routing_model.dep import *
from routing_model.utils import *
from routing_model.layers import reinforce_loss, value_loss
from routing_model._learner  import AttentionLearner
from routing_model._args import parse_args, write_config_file

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

from tensorboard_logger import Logger as TbLogger

import time
import os
from itertools import chain
import numpy as np
import gc

def train_epoch(args, data, Environment, env_params, bl_wrapped_learner, optim, device, ep, **kwargs):
    if args.train_mode == 'value':
        assert 'policy_baseline' in kwargs, 'Value training mode needs policy model to be passed'
        policy_bl_wrapped_learner = kwargs.get('policy_baseline')
        policy_bl_wrapped_learner.learner.eval()
    bl_wrapped_learner.learner.train()
    loader = DataLoader(data, args.batch_size, True)

    if not args.no_tensorboard:
        tb_logger = kwargs.get('tb_logger')
    step = ep * args.iter_count
    
    ep_loss = []
    ep_bl_loss = []
    ep_prob = []
    ep_val = []
    ep_bl = []
    ep_norm = []
    with tqdm(loader, desc = "Ep.#{: >3d}/{: <3d}".format(ep+1, args.epoch_count)) as progress:
        for minibatch in progress:
            if data.cust_mask is None:
                vehs, custs, mask = minibatch[0].to(device), minibatch[1].to(device), None
            else:
                vehs, custs, mask = minibatch[0].to(device), minibatch[1].to(device), minibatch[2].to(device)

            dyna = Environment(data, vehs, custs, mask, *env_params)
            if args.train_mode == 'policy':
                _, logps, rewards, bl_vals = bl_wrapped_learner(dyna)
                loss, bl_loss = reinforce_loss(logps, rewards, bl_vals)
                prob = torch.stack(logps).sum(0).exp().mean()
                val = rewards.mean()            
                bl = bl_vals.mean()
                optim.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    grad_norm = clip_grad_norm_(chain.from_iterable(grp["params"] for grp in optim.param_groups),
                            args.max_grad_norm)
                optim.step()
                progress.set_postfix_str("l={:.4g} vl={:9.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(
                    loss, bl_loss, prob, val, bl, grad_norm))
            elif args.train_mode == 'value':
                with torch.no_grad(): _, logps, rewards, _ = policy_bl_wrapped_learner(dyna, greedy=True)
                _, _, _, bl_vals = bl_wrapped_learner(dyna)
                loss, bl_loss = reinforce_loss(logps, rewards, bl_vals)
                prob = torch.stack(logps).sum(0).exp().mean()
                val = rewards.mean()            
                bl = bl_vals.mean()
                optim.zero_grad()
                bl_loss.backward()
                if args.max_grad_norm is not None:
                    grad_norm = clip_grad_norm_(chain.from_iterable(grp["params"] for grp in optim.param_groups),
                            args.max_grad_norm)
                optim.step()
                progress.set_postfix_str("l={:.4g} vl={:9.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(
                    loss, bl_loss, prob, val, bl, grad_norm))
                
            if not args.no_tensorboard:
                step += 1
                log_values(tb_logger, step, loss, bl_loss, prob, val, bl, grad_norm)

            ep_loss.append(loss)
            ep_bl_loss.append(bl_loss)
            ep_prob.append(prob)
            ep_val.append(val)
            ep_bl.append(bl)
            ep_norm.append(grad_norm)

            # del minibatch, vehs, custs, mask
            # del dyna, logps, rewards, bl_vals
            # del loss, bl_loss, prob, val, bl, grad_norm, 

            # print(torch.cuda.memory_summary('cuda:3'))
            # torch.cuda.empty_cache()
            # gc.collect()

        ep_loss = torch.stack(ep_loss)
        ep_bl_loss = torch.stack(ep_bl_loss)
        ep_prob = torch.stack(ep_prob)
        ep_val = torch.stack(ep_val)
        ep_bl = torch.stack(ep_bl)
        ep_norm = torch.stack(ep_norm)

    return ep_loss.mean().item(), ep_bl_loss.mean().item(), ep_prob.mean().item(), ep_val.mean().item(), ep_bl.mean().item(), ep_norm.mean().item(),\
            ep_loss.std().item(), ep_bl_loss.std().item(), ep_val.std().item(), ep_bl.std().item()


def test_epoch(args, test_env, learner, bl_wrapped_learner, ref_costs, **kwargs):
    bl_wrapped_learner.learner.eval()
    if args.train_mode == 'value':
        policy_bl_wrapped_learner = kwargs.get('policy_baseline')
        policy_bl_wrapped_learner.learner.eval()

    with torch.no_grad():
        if args.problem_type[0] == "s":
            losses = []
            bl_losses = []
            costs = []
            for _ in range(100):
                if args.train_mode == 'policy':
                    actions, logps, rewards, bl_vals = bl_wrapped_learner(test_env, greedy=True)
                    loss, bl_loss = reinforce_loss(logps, rewards, bl_vals)
                elif args.train_mode == 'value':
                    actions, logps, rewards, _ = policy_bl_wrapped_learner(test_env, greedy=True)
                    _, _, _, bl_vals = bl_wrapped_learner(test_env)
                    loss, bl_loss = reinforce_loss(logps, rewards, bl_vals)
                costs.append( rewards.mean() )
                losses.append(loss)
                bl_losses.append(bl_loss)
            costs = torch.stack(costs)
            losses = torch.stack(losses)
            bl_losses = torch.stack(bl_losses)
        else:
            _, _, rs = learner(test_env)
            costs = -torch.stack(rs).sum(dim = 0).squeeze(-1)

    gap = (costs.to(ref_costs.device) / ref_costs - 1).mean() if ref_costs is not None else 0
    print("Cost on test dataset: {:5.2f} +- {:5.2f} ({:.2%})".format(costs.mean(), costs.std(), gap))
    return costs.mean().item(), losses.mean().item(), bl_losses.mean().item(), gap, \
        losses.std().item(), bl_losses.std().item()

def main(args):
    if args.verbose:
        verbose_print = print
    else:
        def verbose_print(*args, **kwargs): pass

    if torch.cuda.is_available() and not args.no_cuda:
        num_of_gpus = torch.cuda.device_count()
        verbose_print("Find {} GPU cores in the device".format(num_of_gpus))
        if args.gpu_no < num_of_gpus:
            device = 'cuda:'+str(args.gpu_no)
            verbose_print("GPU:{} is selected".format(args.gpu_no))
        else:
            raise ValueError('GPU: {} is not available in the device'.format(args.gpu_no))
    else:
        device = 'cpu'

    dev = torch.device(device)
    if args.rng_seed is not None:
        torch.manual_seed(args.rng_seed)

    # PROBLEM
    Dataset = {
            "vrp": VRP_Dataset,
            "vrptw": VRPTW_Dataset,
            "svrptw": VRPTW_Dataset,
            "sdvrptw": SDVRPTW_Dataset
            }.get(args.problem_type)

    gen_params = [
            args.customers_count,
            args.vehicles_count,
            args.veh_capa_range,
            args.veh_speed_range,
            args.min_cust_count,
            args.max_cust_count,
            args.loc_range,
            args.dem_range,
            args.rew_range
            ]
    if args.problem_type != "vrp":
        gen_params.extend( [args.horizon, args.dur_range, args.tw_ratio, args.tw_range] )
    if args.problem_type == "sdvrptw":
        gen_params.extend( [args.deg_of_dyna, args.appear_early_ratio] )

    # TRAIN DATA
    verbose_print("Generating {} {} samples of training data...".format(
        args.iter_count * args.batch_size, args.problem_type.upper()),
        end = " ", flush = True)
    train_data = Dataset.generate(
            args.iter_count * args.batch_size,
            *gen_params
            )
    train_data.normalize()
    # out_dir = "train_data/s_cvrptw_n1-100m1"
    # os.save(train_data, os.path.join(out_dir, "norm_data.pyth"))
    verbose_print("Done.")

    # TEST DATA AND COST REFERENCE
    verbose_print("Generating {} {} samples of test data...".format(
        args.test_batch_size, args.problem_type.upper()),
        end = " ", flush = True)
    test_data = Dataset.generate(
            args.test_batch_size,
            *gen_params
            )
    verbose_print("Done.")

    ORTOOLS_ENABLED = False

    if ORTOOLS_ENABLED:
        ref_routes = ort_solve(test_data)
    elif LKH_ENABLED:
        ref_routes = lkh_solve(test_data)
    else:
        ref_routes = None
        print("Warning! No external solver found to compute gaps for test.")
    test_data.normalize()

    # ENVIRONMENT
    Environment = {
            "vrp": VRP_Environment,
            "vrptw": VRPTW_Environment,
            "svrptw": SVRPTW_Environment,
            "sdvrptw": SDVRPTW_Environment
            }.get(args.problem_type)
    env_params = [args.pending_cost]
    if args.problem_type != "vrp":
        env_params.append(args.late_cost)
        if args.problem_type != "vrptw":
            env_params.extend( [args.speed_var, args.late_prob, args.slow_down, args.late_var] )
    test_env = Environment(test_data, None, None, None, *env_params)

    if ref_routes is not None:
        ref_costs = eval_apriori_routes(test_env, ref_routes, 100 if args.problem_type[0] == 's' else 1)
        print("Reference cost on test dataset {:5.2f} +- {:5.2f}".format(ref_costs.mean(), ref_costs.std()))
    test_env.nodes = test_env.nodes.to(dev)
    if test_env.init_cust_mask is not None:
        test_env.init_cust_mask = test_env.init_cust_mask.to(dev)

    # MODEL
    verbose_print("Initializing attention model...",
        end = " ", flush = True)
    learner = AttentionLearner(
            Dataset.CUST_FEAT_SIZE,
            Environment.VEH_STATE_SIZE,
            args.model_size,
            args.layer_count,
            args.head_count,
            args.ff_size,
            args.tanh_xplor
            )
    learner.to(dev)
    verbose_print("Done.")

    # BASELINE
    verbose_print("Initializing '{}' baseline...".format(
        args.baseline_type),
        end = " ", flush = True)
    if args.baseline_type == "none":
        baseline = NoBaseline(learner)
    elif args.baseline_type == "nearnb":
        baseline = NearestNeighbourBaseline(learner, args.loss_use_cumul)
    elif args.baseline_type == "rollout":
        args.loss_use_cumul = True
        baseline = RolloutBaseline(learner, args.rollout_count, args.rollout_threshold)
    elif args.baseline_type == "critic":
        baseline = CriticBaseline(learner, args.customers_count, args.critic_use_qval, args.loss_use_cumul)
    baseline.to(dev)
    verbose_print("Done.")

    # OPTIMIZER AND LR SCHEDULER
    verbose_print("Initializing Adam optimizer...",
        end = " ", flush = True)
    lr_sched = None
    if args.baseline_type == "critic":
        optim = Adam([
            {"params": learner.parameters(), "lr": args.learning_rate},
            {"params": baseline.parameters(), "lr": args.critic_rate}
            ])
        if args.rate_decay is not None:
            critic_decay = args.rate_decay if args.critic_decay is None else args.critic_decay
            lr_sched = LambdaLR(optim,[
                lambda ep: args.learning_rate * args.rate_decay**ep,
                lambda ep: args.critic_rate * critic_decay**ep
                ])
    else:
        optim = Adam(learner.parameters(), args.learning_rate)
        if args.rate_decay is not None:
            lr_sched = LambdaLR(optim, lambda ep: args.learning_rate * args.rate_decay**ep)
    verbose_print("Done.")

    # CHECKPOINTING
    verbose_print("Creating output dir...",
        end = " ", flush = True)
    args.output_dir = "vrp_output/{}n{}m{}_{}".format(
            args.problem_type.upper(),
            args.customers_count,
            args.vehicles_count,
            time.strftime("%y%m%d-%H%M")
            ) if args.output_dir is None else args.output_dir
    os.makedirs(args.output_dir, exist_ok = True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    verbose_print("'{}' created.".format(args.output_dir))

    if args.resume_state is None:
        start_ep = 0
    else:
        start_ep = load_checkpoint(args, learner, baseline=None, lr_sched=lr_sched)
    
    # TENSORBOARD LOGGING
    verbose_print("Creating log dir...",
    end = " ", flush = True)
    args.log_dir = "log/{}n{}m{}_{}".format(
            args.problem_type.upper(),
            args.customers_count,
            args.vehicles_count,
            time.strftime("%y%m%d-%H%M")
            ) if args.log_dir is None else args.log_dir
    os.makedirs(args.output_dir, exist_ok = True)

    tb_logger = None    
    if not args.no_tensorboard:
        tb_logger = TbLogger(args.log_dir)
        verbose_print("'{}' created.".format(args.log_dir))

    # SEPARATE POLICY MODEL FOR VALUE TRAINING MODE
    if args.train_mode == 'value':
        verbose_print("Initialize the policy network separately...",
            end = " ", flush = True)
        if args.baseline_type == "critic":
            verbose_print("Initializing policy attention model...",
                end = " ", flush = True)
            # ATTENTION MODEL
            policy_learner = AttentionLearner(
                    Dataset.CUST_FEAT_SIZE,
                    Environment.VEH_STATE_SIZE,
                    args.model_size,
                    args.layer_count,
                    args.head_count,
                    args.ff_size,
                    args.tanh_xplor
                    )
            policy_learner.to(dev)
            verbose_print("Done.")
            # BASELINE
            verbose_print("Initializing '{}' baseline for policy model...".format(
                args.baseline_type),
                end = " ", flush = True)
            policy_baseline = CriticBaseline(policy_learner, args.customers_count, args.critic_use_qval, args.loss_use_cumul)
            policy_baseline.to(dev)
            verbose_print("Done.")
        else: raise ValueError("{} baseline don't support value training mode".format(args.baseline_type))
        # CHECKPOINTING
        if args.resume_state is None:
            start_ep = 0
        else:
            start_ep = load_checkpoint(args, policy_learner, None)


    verbose_print("Running...")
    train_stats = []
    test_stats = []
    try:
        for ep in range(start_ep, args.epoch_count):
            if args.train_mode == 'policy':
                train_stats.append( train_epoch(args, train_data, Environment, env_params, baseline, optim, dev, ep,
                                                tb_logger=tb_logger) )
                test_stats.append( test_epoch(args, test_env, learner, baseline, ref_costs=None) )
            elif args.train_mode == 'value':
                train_stats.append( train_epoch(args, train_data, Environment, env_params, baseline, optim, dev, ep,\
                    policy_baseline = policy_baseline, tb_logger=tb_logger) )
                test_stats.append( test_epoch(args, test_env, learner, baseline, ref_costs=None, policy_baseline = policy_baseline) )
            else:
                raise ValueError(f'do not support training model: {args.train_mode}')

            if args.rate_decay is not None:
                lr_sched.step()
            if args.pend_cost_growth is not None:
                env_params[0] *= args.pend_cost_growth
            if args.late_cost_growth is not None:
                env_params[1] *= args.late_cost_growth
            if args.grad_norm_decay is not None:
                args.max_grad_norm *= args.grad_norm_decay

            if (ep+1) % args.checkpoint_period == 0:
                save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
                

    except KeyboardInterrupt:
        save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
    finally:
        export_train_test_stats(args, start_ep, train_stats, test_stats)


if __name__ == "__main__":
    main(parse_args())
