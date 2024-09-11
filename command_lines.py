

# Run CBBA task allocation code

python CBBA_TaMain.py --speed-var 0.00 --late-prob 0.0

python vrp_main.py --gpu-no 1 --customers-count 10 --vehicles-count 1 --min-cust-count 0 --max-cust-count 10 --speed-var 0.2 --late-prob 0.0 --epoch-count 100 --train-mode policy
python vrp_main.py --customers-count 10 --vehicles-count 1 --min-cust-count 0 --max-cust-count 10 --speed-var 0.0 --late-prob 0.0 --epoch-count 150 --train-mode value --resume-state 'vrp_output/SVRPTWn10m1_221015-1508/chkpt_ep100.pyth'

# MARL
python vrp_main.py --gpu-no 1 --customers-count 10 --vehicles-count 2 --speed-var 0.00 --late-prob 0.0 --epoch-count 100 --train-mode policy
python vrp_main.py --gpu-no 2 --customers-count 20 --vehicles-count 4 --speed-var 0.00 --late-prob 0.0 --epoch-count 100 --train-mode policy
python vrp_main.py --gpu-no 3 --customers-count 50 --vehicles-count 10 --speed-var 0.00 --late-prob 0.0 --epoch-count 100 --train-mode policy

python CBBA_TaMain.py --speed-var 0.2 --late-prob 0.0 



python vrp_main.py --customers-count 10 --vehicles-count 1 --min-cust-count 0 --max-cust-count 10 --speed-var 0.0 --late-prob 0.0 --epoch-count 150 --train-mode value --resume-state 'vrp_output/SVRPTWn10m1_230925-0012/chkpt_ep100.pyth'


python vrp_main.py --customers-count 10 --vehicles-count 2 --speed-var 0.00 --late-prob 0.0 --epoch-count 100 --train-mode policy

python -m tensorboard.main --logdir='C:\Users\s313488\OneDrive - Cranfield University\1-FanFile\7-Repos\1-PhD_multi_uav_mission_planning\Deep-auction\log'  --host localhost --port 8088


python -m tensorboard.main --logdir='/Users/ruifan/Library/CloudStorage/OneDrive-CranfieldUniversity/1-FanFile/7-Repos/1-PhD_multi_uav_mission_planning/Deep-auction/log'  --host localhost --port 8088

python vrp_main.py --customers-count 10 --vehicles-count 1 --min-cust-count 0 --max-cust-count 10 --speed-var 0.2 --late-prob 0.0 --epoch-count 150 --train-mode value --resume-state 'vrp_output/SVRPTWn10m1_240427-1413/chkpt_ep100.pyth'

python vrp_main.py --customers-count 10 --vehicles-count 2 --speed-var 0.00 --late-prob 0.0 --epoch-count 100 --train-mode policy
