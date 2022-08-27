import torch
import os.path

def save_checkpoint(args, ep, learner, optim, baseline = None, lr_sched = None):
    checkpoint = {
            "ep": ep,
            "model": learner.state_dict(),
            "optim": optim.state_dict()
            }
    if args.rate_decay is not None:
        checkpoint["lr_sched"] = lr_sched.state_dict()
    if args.baseline_type == "critic":
        checkpoint["critic"] = baseline.state_dict()
    torch.save(checkpoint, os.path.join(args.output_dir, "chkpt_ep{}.pyth".format(ep+1)))


def load_checkpoint(args, learner, optim, baseline = None, lr_sched = None):
    checkpoint = torch.load(args.resume_state)
    learner.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    if args.rate_decay is not None:
        lr_sched.load_state_dict(checkpoint["lr_sched"])
    if args.baseline_type == "critic":
        baseline.load_state_dict(checkpoint["critic"])
    return checkpoint["ep"]
