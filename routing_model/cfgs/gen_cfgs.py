import json
import random

pb_types = ["vrptw", "svrptw", "sdvrptw"]
counts = ((10,2), (20,4), (50,10))
runs = 5
epochs = 20

with open("./cfgs/launch_all.sh", 'w') as sh_f:
    sh_f.write('export MPLBACKEND="Agg"\n\n')
    for pb in pb_types:
        for n,m in counts:
            for r in range(runs):
                json_fpath = "./cfgs/{}_n{}m{}_{}.json".format(pb,n,m,r)
                with open(json_fpath, 'w') as json_f:
                    json.dump({
                        "rng_seed": random.randint(1e8,1e9-1),
                        "problem_type": pb,
                        "customers_count": n,
                        "vehicles_count": m,
                        "epoch_count": epochs,
                        "iter_count": 2500,
                        "baseline_type": "critic",
                        "plot_period": epochs,
                        "plot_select": "best"
                        }, json_f, indent = 4)
                    json_f.write('\n')
                sh_f.write("./script/train.py -f {}\n".format(json_fpath))
            sh_f.write('\n')
