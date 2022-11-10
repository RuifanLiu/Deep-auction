

# Run CBBA task allocation code

python CBBA_TaMain.py --speed-var 0.00 --late-prob 0.0
python CBBA_TaMain.py --customers-count 50 --vehicles-count 10 --speed-var 0.2 --late-prob 0.0 --epoch-count 50 --train-mode policy

python vrp_main.py --gpu-no 1 --customers-count 10 --vehicles-count 1 --min-cust-count 0 --max-cust-count 10 --speed-var 0.0 --late-prob 0.0 --epoch-count 100 --train-mode policy
python vrp_main.py --customers-count 10 --vehicles-count 1 --min-cust-count 0 --max-cust-count 10 --speed-var 0.0 --late-prob 0.0 --epoch-count 150 --train-mode value --resume-state 'vrp_output/SVRPTWn10m1_221015-1508/chkpt_ep100.pyth'
# MARL
python vrp_main.py --gpu-no 1 --customers-count 10 --vehicles-count 2 --speed-var 0.00 --late-prob 0.0 --epoch-count 100 --train-mode policy
python vrp_main.py --gpu-no 2 --customers-count 20 --vehicles-count 4 --speed-var 0.00 --late-prob 0.0 --epoch-count 100 --train-mode policy
python vrp_main.py --gpu-no 3 --customers-count 50 --vehicles-count 10 --speed-var 0.00 --late-prob 0.0 --epoch-count 100 --train-mode policy

python CBBA_TaMain.py --speed-var 0.00 --late-prob 0.0 
