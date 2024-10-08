from argparse import ArgumentParser
import sys
import json

CONFIG_FILE = None
VERBOSE = True
NO_CUDA = False
SEED = None

PROBLEM = "svrptw"
CUST_RANGE = [100] 
VEH_RANGE = [20]
VALID_DATASET = 'data_sample10_stw/'
ITERATION = 10

# stochastic models with speed var 0.2
VALUE_MODEL = 'vrp_output/SVRPTWn10m1_221011-2044/chkpt_ep150.pyth'
DRL_MODEL = 'vrp_output/SVRPTWn10m1_221011-1625/chkpt_ep100.pyth'
# deterministic models
# VALUE_MODEL = 'vrp_output/SVRPTWn10m1_221016-1509/chkpt_ep150.pyth'
# DRL_MODEL = 'vrp_output/SVRPTWn10m1_221015-1508/chkpt_ep100.pyth'


PEND_COST = 1
PEND_GROWTH = None
# LATE_DISCOUNT = 0.9
LATE_COST = 4
LATE_GROWTH = None
SPEED_VAR = 0.0
LATE_PROB = 0.0
SLOW_DOWN = 0.2
LATE_VAR = 0.2

VALID_BATCH_SIZE = 1
OUTPUT_DIR = None


def write_config_file(args, output_file):
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent = 4)


def parse_args(argv = None):
    parser = ArgumentParser()

    parser.add_argument("--config-file", "-f", type = str, default = CONFIG_FILE)
    parser.add_argument("--verbose", "-v", action = "store_true", default = VERBOSE)
    parser.add_argument("--no-cuda", action = "store_true", default = NO_CUDA)
    parser.add_argument("--rng-seed", type = int, default = SEED)

    group = parser.add_argument_group("simulation parameters")
    group.add_argument("--customers-range", type = float, nargs = '*', default = CUST_RANGE)

    group.add_argument("--vehicles-range", type = float, nargs = '*', default = VEH_RANGE)
    group.add_argument("--problem-type", "-p", type = str,
            choices = ["vrp", "vrptw", "svrptw", "sdvrptw"], default = PROBLEM)
    group.add_argument("--valid-dataset",  type = str, default = VALID_DATASET)
    parser.add_argument("--sim-iteartion", type = int, default = ITERATION)
    
    group = parser.add_argument_group("Loaded model")
    group.add_argument("--value-model", type = str, default = VALUE_MODEL)
    group.add_argument("--drl-model", type = str, default = DRL_MODEL)

    group = parser.add_argument_group("VRP Environment parameters")
    group.add_argument("--pending-cost", type = float, default = PEND_COST)
    group.add_argument("--pend-cost-growth", type = float, default = PEND_GROWTH)
    group.add_argument("--late-cost", type = float, default = LATE_COST)
    group.add_argument("--late-cost-growth", type = float, default = LATE_GROWTH)
    group.add_argument("--speed-var", type = float, default = SPEED_VAR)
    group.add_argument("--late-prob", type = float, default = LATE_PROB)
    group.add_argument("--slow-down", type = float, default = SLOW_DOWN)
    group.add_argument("--late-var", type = float, default = LATE_VAR)

    group = parser.add_argument_group("Testing parameters")
    group.add_argument("--valid-batch-size", type = int, default = VALID_BATCH_SIZE)
    group.add_argument("--output-dir", "-o", type = str, default = OUTPUT_DIR)

    group = parser.add_argument_group("Methods")
    parser.add_argument("--dnn-cbba", action = "store_false")
    parser.add_argument("--mdp-cbba", action = "store_true")
    parser.add_argument("--cbba", action = "store_false")
    parser.add_argument("--robust_cbba", action = "store_true")

    args = parser.parse_args(argv)
    if args.config_file is not None:
        with open(args.config_file) as f:
            parser.set_defaults(**json.load(f))

    return parser.parse_args(argv)
