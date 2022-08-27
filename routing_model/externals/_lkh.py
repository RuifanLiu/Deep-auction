from routing_model.dep import LKH_ENABLED, LKH_BIN
from routing_model.dep import tqdm

from multiprocessing import Pool
import subprocess
import tempfile
import os.path

def _call_lkh(nodes, veh_count, veh_capa, prefix = "/tmp/mardan_lkh0"):
    tsp_path = "{}.tsp".format(prefix)
    with open(tsp_path, 'w') as tsp_f:
        tsp_f.write("NAME: temp\n")
        tsp_f.write("TYPE: {}\n".format("CVRPTW" if nodes.size(1) > 3 else "CVRP"))
        tsp_f.write("DIMENSION: {}\n".format(nodes.size(0)))
        tsp_f.write("VEHICLES: {}\n".format(veh_count))
        tsp_f.write("CAPACITY: {}\n".format(veh_capa))
        tsp_f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        tsp_f.write("EDGE_WEIGHT_FORMAT: FUNCTION\n")
        tsp_f.write("NODE_COORD_TYPE: TWOD_COORDS\n")
        tsp_f.write("NODE_COORD_SECTION\n{}\n".format(
            "\n".join("{} {:.0f} {:.0f}".format(j,x,y) for j,(x,y) in enumerate(nodes[:,:2], start = 1))
            ))
        tsp_f.write("DEPOT_SECTION\n1\n-1\n")
        tsp_f.write("DEMAND_SECTION\n{}\n".format(
            "\n".join("{} {:.0f}".format(j,q) for j,q in enumerate(nodes[:,2], start = 1))
            ))
        if nodes.size(1) > 3:
            tsp_f.write("SERVICE_TIME_SECTION\n{}\n".format(
                "\n".join("{} {:.0f}".format(j,d) for j,d in enumerate(nodes[:,5], start = 1))
                ))
            tsp_f.write("TIME_WINDOW_SECTION\n{}\n".format(
                "\n".join("{} {:.0f} {:.0f}".format(j,e,l) for j,(e,l) in enumerate(nodes[:,3:5], start = 1))
                ))

    par_path = "{}.par".format(prefix)
    tr_path = "{}.tour".format(prefix)
    with open(par_path, "w") as par_f:
        par_f.write("SPECIAL\n")
        par_f.write("PROBLEM_FILE = {}\n".format(tsp_path))
        par_f.write("MTSP_SOLUTION_FILE = {}\n".format(tr_path))
        par_f.write("MAX_TRIALS = 4000\n")
        par_f.write("RUNS = 2\n")

    subprocess.run([LKH_BIN, par_path], stdout = subprocess.DEVNULL)

    with open(tr_path, 'r') as tr_f:
        lines = tr_f.readlines()
    return [[int(j)-1 for j in l.split('(',1)[0].split()[1:]] for l in lines[2:]]


def lkh_solve(data):
    with Pool() as p:
        with tqdm(desc = "Calling LKH3", total = data.batch_size) as pbar:
            with tempfile.TemporaryDirectory(prefix = "mardan_lkh") as tmp_dir:
                results = [p.apply_async(_call_lkh, (nodes, data.veh_count, data.veh_capa,
                    os.path.join(tmp_dir, str(b))), callback = lambda _:pbar.update())
                    for b,nodes in enumerate(data.nodes_gen())]
                routes = [res.get() for res in results]
    return routes
