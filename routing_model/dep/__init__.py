try:
    from tqdm import tqdm
    TQDM_ENABLED = True
except ImportError:
    class tqdm:
        def __init__(self, iterable, total = -1, desc = ""):
            self.iterable = iterable
            self.total = total if total > 0 else len(iterable)
            self.desc = desc
        def __iter__(self):
            print("\r{}  0% ...".format(self.desc), end = '', flush = True)
            for i,elem in enumerate(self.iterable):
                yield elem
                print("\r{} {: 4.0%} ...".format(self.desc, (i+1) / self.total), end = '', flush = True)
            print(" Done!")
    TQDM_ENABLED = False

try:
    import matplotlib
    from matplotlib import pyplot
    matplotlib.rcParams["backend"] = "Agg"
    MPL_ENABLED = True
except ImportError:
    matplotlib = None
    pyplot = None
    MPL_ENABLED = False

try:
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver import routing_enums_pb2
    ORTOOLS_ENABLED = True
except ImportError:
    pywrapcp = None
    routing_enums_pb2 = None
    ORTOOLS_ENABLED = False

import os as _os
for cand in [
        "./bin/LKH",
        _os.path.join(_os.environ.get("HOME", "~"), "LKH-3.0.5/LKH"),
        "/usr/local/bin/LKH",
        "/usr/bin/LKH"
        ]:
    if _os.path.isfile(cand) and _os.access(cand, _os.X_OK):
        LKH_ENABLED = True
        LKH_BIN = cand
        break
else:
    LKH_ENABLED = False
    LKH_BIN = None

try:
    from scipy.stats import ttest_rel
    SCIPY_ENABLED = True
except ImportError:
    ttest_rel = None
    SCIPY_ENABLED = False
