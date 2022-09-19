from argparse import ArgumentParser
import sys
import json

CONFIG_FILE = None
VERBOSE = True
NO_CUDA = False
SEED = None

PROBLEM = "svrptw"
CUST_COUNT = 10
VEH_COUNT = 1
VEH_CAPA_RANGE = (100, 100)
VEH_SPEED_RANGE = (1, 1)
HORIZON = 480
MIN_CUST_COUNT = 0
LOC_RANGE = (0,101)
DEM_RANGE = (0,0)
REW_RANGE = (1,1)
DUR_RANGE = (10,31)
TW_RATIO = (0.25,0.5,0.75,1.0)
TW_RANGE = (30,91)
DEG_OF_DYN = (0.1,0.25,0.5,0.75)
APPEAR_EARLY_RATIO = (0.0,0.5,0.75,1.0)

PEND_COST = 2
PEND_GROWTH = None
# LATE_DISCOUNT = 0.9
LATE_COST = 1
LATE_GROWTH = None
SPEED_VAR = 0.2
LATE_PROB = 0.05
SLOW_DOWN = 0.2
LATE_VAR = 0.2

MODEL_SIZE = 128
LAYER_COUNT = 3
HEAD_COUNT = 8
FF_SIZE = 512
TANH_XPLOR = 10

EPOCH_COUNT = 70
ITER_COUNT = 1000
MINIBATCH_SIZE = 512
BASE_LR = 0.0001
LR_DECAY = None
MAX_GRAD_NORM = 2
GRAD_NORM_DECAY = None
LOSS_USE_CUMUL = True

BASELINE = "critic"
ROLLOUT_COUNT = 3
ROLLOUT_THRESHOLD = 0.05
CRITIC_USE_QVAL = False
CRITIC_LR = 0.001
CRITIC_DECAY = None

TEST_BATCH_SIZE = 1280

OUTPUT_DIR = None
RESUME_STATE = 'vrp_output/SVRPTWn10m1_220918-2341/chkpt_ep50.pyth'
CHECKPOINT_PERIOD = 5
TRAIN_MODE = 'value'


def write_config_file(args, output_file):
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent = 4)


def parse_args(argv = None):
    parser = ArgumentParser()

    parser.add_argument("--config-file", "-f", type = str, default = CONFIG_FILE)
    parser.add_argument("--verbose", "-v", action = "store_true", default = VERBOSE)
    parser.add_argument("--no-cuda", action = "store_true", default = NO_CUDA)
    parser.add_argument("--rng-seed", type = int, default = SEED)

    group = parser.add_argument_group("Data generation parameters")
    group.add_argument("--problem-type", "-p", type = str,
            choices = ["vrp", "vrptw", "svrptw", "sdvrptw"], default = PROBLEM)
    group.add_argument("--customers-count", "-n", type = int, default = CUST_COUNT)
    group.add_argument("--vehicles-count", "-m", type = int, default = VEH_COUNT)
    group.add_argument("--veh-capa-range", type = int, default = VEH_CAPA_RANGE)
    group.add_argument("--veh-speed-range", type = int, default = VEH_SPEED_RANGE)
    group.add_argument("--horizon", type = int, default = HORIZON)
    group.add_argument("--min-cust-count", type = int, default = MIN_CUST_COUNT)
    group.add_argument("--loc-range", type = int, nargs = 2, default = LOC_RANGE)
    group.add_argument("--dem-range", type = int, nargs = 2, default = DEM_RANGE)
    group.add_argument("--rew-range", type = int, nargs = 2, default = REW_RANGE)
    group.add_argument("--dur-range", type = int, nargs = 2, default = DUR_RANGE)
    group.add_argument("--tw-ratio", type = float, nargs = '*', default = TW_RATIO)
    group.add_argument("--tw-range", type = int, nargs = 2, default = TW_RANGE)
    group.add_argument("--deg-of-dyna", type = float, nargs = '*', default = DEG_OF_DYN)
    group.add_argument("--appear-early-ratio", type = float, nargs = '*', default = APPEAR_EARLY_RATIO)

    group = parser.add_argument_group("VRP Environment parameters")
    group.add_argument("--pending-cost", type = float, default = PEND_COST)
    group.add_argument("--pend-cost-growth", type = float, default = PEND_GROWTH)
    group.add_argument("--late-cost", type = float, default = LATE_COST)
    group.add_argument("--late-cost-growth", type = float, default = LATE_GROWTH)
    group.add_argument("--speed-var", type = float, default = SPEED_VAR)
    group.add_argument("--late-prob", type = float, default = LATE_PROB)
    group.add_argument("--slow-down", type = float, default = SLOW_DOWN)
    group.add_argument("--late-var", type = float, default = LATE_VAR)

    group = parser.add_argument_group("Model parameters")
    group.add_argument("--model-size", "-s", type = int, default = MODEL_SIZE)
    group.add_argument("--layer-count", type = int, default = LAYER_COUNT)
    group.add_argument("--head-count", type = int, default = HEAD_COUNT)
    group.add_argument("--ff-size", type = int, default = FF_SIZE)
    group.add_argument("--tanh-xplor", type = float, default = TANH_XPLOR)

    group = parser.add_argument_group("Training parameters")
    group.add_argument("--epoch-count", "-e", type = int, default = EPOCH_COUNT)
    group.add_argument("--iter-count", "-i", type = int, default = ITER_COUNT)
    group.add_argument("--batch-size", "-b", type = int, default = MINIBATCH_SIZE)
    group.add_argument("--learning-rate", "-r", type = float, default = BASE_LR)
    group.add_argument("--rate-decay", "-d", type = float, default = LR_DECAY)
    group.add_argument("--max-grad-norm", type = float, default = MAX_GRAD_NORM)
    group.add_argument("--grad-norm-decay", type = float, default = GRAD_NORM_DECAY)
    group.add_argument("--loss-use-cumul", action = "store_true", default = LOSS_USE_CUMUL)

    group = parser.add_argument_group("Baselines parameters")
    group.add_argument("--baseline-type", type = str,
            choices = ["none", "nearnb", "rollout", "critic"], default = BASELINE)
    group.add_argument("--rollout-count", type = int, default = ROLLOUT_COUNT)
    group.add_argument("--rollout-threshold", type = float, default = ROLLOUT_THRESHOLD)
    group.add_argument("--critic-use-qval", action = "store_true", default = CRITIC_USE_QVAL)
    group.add_argument("--critic-rate", type = float, default = CRITIC_LR)
    group.add_argument("--critic-decay", type = float, default = CRITIC_DECAY)

    group = parser.add_argument_group("Testing parameters")
    group.add_argument("--test-batch-size", type = int, default = TEST_BATCH_SIZE)

    group = parser.add_argument_group("Checkpointing")
    group.add_argument("--output-dir", "-o", type = str, default = OUTPUT_DIR)
    group.add_argument("--checkpoint-period", "-c", type = int, default = CHECKPOINT_PERIOD)
    group.add_argument("--resume-state", type = str, default = RESUME_STATE)
    group.add_argument("--train-mode", type = str, choices=['policy', 'value'], default = TRAIN_MODE)


    args = parser.parse_args(argv)
    if args.config_file is not None:
        with open(args.config_file) as f:
            parser.set_defaults(**json.load(f))

    return parser.parse_args(argv)
