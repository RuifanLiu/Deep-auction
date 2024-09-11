# Deep Auction

M-MDP for multi-agent stochastic planning is decomposed into a series of individual MDPs and then solved by Auctioning, which we called Deep Auction.

## Paer 

For more details, please see our pre-printed paper:

Liu, R., Shin, H.S., Yan, B. and Tsourdos, A., 2022. An auction-based coordination strategy for task-constrained multi-agent stochastic planning with submodular rewards. arXiv preprint arXiv:2212.14624. [ArXiv Link](An auction-based coordination strategy for task-constrained multi-agent stochastic planning with submodular rewards)


## Dependencies
- Python
- NumPy
- SciPy
- PyTorch
- tqdm
- tensorboard_logger
- Matplotlib (optional, only for plotting)


## Usage
### Training
The training algorithm consists of 2 phases.
#### Phase 1 Policy training
For training VRP instances with 1-10 customers:
```bash
python vrp_main.py --customers-count 10 --vehicles-count 1 --min-cust-count 0 --max-cust-count 10 --speed-var 0.2 --late-prob 0.0 --epoch-count 100 --train-mode policy
```
### Phase 2 Value network training
Fix the actor network, only the critic network is updated by minimizing the MSE:
```bash
python vrp_main.py --customers-count 10 --vehicles-count 1 --min-cust-count 0 --max-cust-count 10 --speed-var 0.2 --late-prob 0.0 --epoch-count 150 --train-mode value --resume-state 'vrp_output/pre_trained_model/chkpt_ep100.pyth'
```

#### Resume training
You can resume a run using a pre-trained model by using --resume:
```bash
python vrp_main.py --customers-count 10 --vehicles-count 1 --min-cust-count 0 --max-cust-count 10 --speed-var 0.0 --late-prob 0.0 --epoch-count 100 --train-mode policy --resume-state 'vrp_output/pre_trained_model/chkpt_ep50.pyth'
```

### Validation

#### Evaluation
To evaluate the performance, we run CBBA_TaMain.py, where different baselines will be called and results are generated:
```bash
python CBBA_TaMain.py --speed-var 0.2 --late-prob 0.0 
````



