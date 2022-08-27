import torch
import pickle


REF_CELL_STR   = r"${}{{{:5.2f} \pm {:5.2f}}}$"
CELL_STR       = r"${}{{{:5.2f} \pm {:5.2f} ({:3.0f}\%)}}$"


def get_mean_std_gap(sample, ref=None, outlier_factor=1.5):
    q1, _ = sample.kthvalue(sample.numel()//4)
    q3, _ = sample.kthvalue(3*sample.numel()//4)
    mask_outlier = (q1 - outlier_factor*(q3-q1) <= sample) \
            & (sample <= q3 + outlier_factor*(q3-q1))
    masked = sample[mask_outlier]
    gap = None if ref is None else 100*(masked / ref[mask_outlier] - 1).mean()
    return masked.mean(), masked.std(), gap



if __name__ == "__main__":
    ############ DET
    print(r"""\begin{tabular}{|c|c|c|c|c|}
\hline
Pb. &\diagbox{Method}{dim.} &$N = 10$ &$N = 20$ &$N = 50$ \\
\hline
\multirow{6}{*}{\rotatebox[origin=c]{90}{CVRP}} %""")
    tab = []
    for n in (10, 20, 50):
        m = n // 5
        dir_path = "./results/{}_n{}m{}/".format("cvrp", n, m)
        ref = torch.load(dir_path + "lkh.pyth", map_location='cpu')["costs"]
        lkh = get_mean_std_gap(ref, ref)
        try:
            ort = get_mean_std_gap(torch.load(dir_path + "ort.pyth", map_location='cpu')["costs"], ref)
        except FileNotFoundError:
            ort = (float("nan"), float("nan"), float("nan"))

        try:
            with open(dir_path + "kool_greedy.pkl", 'rb') as f:
                res, _ = pickle.load(f)
                am_g, _, _ = zip(*res)
                am_g = get_mean_std_gap(torch.tensor(am_g), ref)
        except FileNotFoundError:
            am_g = (float("nan"), float("nan"), float("nan"))
        try:
            with open(dir_path + "kool_sample100.pkl", 'rb') as f:
                res, _ = pickle.load(f)
                am_s, _, _ = zip(*res)
                am_s = get_mean_std_gap(torch.tensor(am_s), ref)
        except FileNotFoundError:
            am_s = (float("nan"), float("nan"), float("nan"))
        try:
            with open(dir_path + "nazari_greedy.pkl", 'rb') as f:
                nco_g = pickle.load(f)["costs"]
                nco_g = get_mean_std_gap(torch.tensor(nco_g), ref)
        except FileNotFoundError:
            nco_g = (float("nan"), float("nan"), float("nan"))
        try:
            with open(dir_path + "nazari_beam_search.pkl", 'rb') as f:
                nco_bs = pickle.load(f)["costs"]
                nco_bs = get_mean_std_gap(torch.tensor(nco_bs), ref)
        except FileNotFoundError:
            nco_bs = (float("nan"), float("nan"), float("nan"))
        mardam_g = get_mean_std_gap(torch.load(dir_path + "mardan_greedy.pyth", map_location='cpu'), ref)
        mardam_s = get_mean_std_gap(torch.load(dir_path + "mardan_sample100.pyth", map_location='cpu'), ref)

        _, best = min((res[2],k) for k, res in enumerate((lkh, ort, am_g, am_s, nco_g, nco_bs, mardam_g, mardam_s)))

        tab.append([REF_CELL_STR.format(r"\boldsymbol" if best==0 else "", *lkh)] \
                + [CELL_STR.format(r"\boldsymbol" if k==best else "", *res)
                for k, res in enumerate((ort, am_g, am_s, nco_g, nco_bs, mardam_g, mardam_s), start=1)])

    tab = [row for row in zip(*tab)]
    print('\n'.join(r"&{} &{} &{} &{} \\".format(mtd.ljust(10), *row)
        for mtd, row in zip(("LKH", "ORTools", "AM (g)", "AM (s)", "NCO (g)", "NCO (bs)", "MARDAM (g)", "MARDAM (s)"), tab)))
    print(r"""\hline
\multirow{6}{*}{\rotatebox[origin=c]{90}{CVRP-TW}} %
& & & & \\""")
    tab = []
    for n in (10, 20, 50):
        m = n // 5
        dir_path = "./results/{}_n{}m{}/".format("cvrptw", n, m)
        ref = torch.load(dir_path + "lkh.pyth", map_location='cpu')["costs"]
        lkh = get_mean_std_gap(ref, ref)
        ort = get_mean_std_gap(torch.load(dir_path + "ort.pyth", map_location='cpu')["costs"], ref)
        mardam_g = get_mean_std_gap(torch.load(dir_path + "mardan_greedy.pyth", map_location='cpu'), ref)
        mardam_s = get_mean_std_gap(torch.load(dir_path + "mardan_sample100.pyth", map_location='cpu'), ref)

        _, best = min((res[2],k) for k, res in enumerate((lkh, ort, mardam_g, mardam_s)))

        tab.append([REF_CELL_STR.format(r"\boldsymbol" if best==0 else "", *lkh)] \
                + [CELL_STR.format(r"\boldsymbol" if best==k else "", *res)
                for k, res in enumerate((ort, mardam_g, mardam_s), start=1)])

    tab = [row for row in zip(*tab)]
    print('\n'.join(r"&{} &{} &{} &{} \\".format(mtd.ljust(10), *row)
        for mtd, row in zip(("LKH", "ORTools", "MARDAM (g)", "MARDAM (s)"), tab)))
    print(r"""& & & & \\
\hline
\end{tabular}""")
    print()

    ############ STOCH
    print(r"""\begin{tabular}{|c|c|c|c|c|}
\hline
$r_\text{slow}$ &\diagbox{Method}{dim.} &$N = 10$ &$N = 20$ &$N=50$ \\
\hline
\multirow{3}{*}{0\%} %""")
    tab = []
    for n in (10, 20, 50):
        m = n // 5
        dir_path = "./results/{}_n{}m{}/".format("s_cvrptw", n, m)
        ref = torch.load(dir_path + "ort.pyth", map_location='cpu')["costs"]
        ort = get_mean_std_gap(ref, ref)
        mardam_g = get_mean_std_gap(torch.load(dir_path + "mardan_greedy.pyth", map_location='cpu'), ref)

        _, best = min((res[2],k) for k, res in enumerate((ort, mardam_g)))

        tab.append([REF_CELL_STR.format(r"\boldsymbol" if best==0 else "", *ort),
            CELL_STR.format(r"\boldsymbol" if best==1 else "", *mardam_g)])

    tab = [row for row in zip(*tab)]
    print(r"& & & & \\")
    print('\n'.join(r"&{} &{} &{} &{} \\".format(mtd.ljust(23), *row)
        for mtd, row in zip(("ORTools (optimistic)", "MARDAM (g)"), tab)))
    print(r"\hline")

    for late_p in (5, 10, 20, 30, 50):
        print(r"\multirow{{3}}{{*}}{{{}\%}} %".format(late_p))
        tab = []
        for n in (10, 20, 50):
            m = n // 5
            dir_path = "./results/{}_n{}m{}/".format("s_cvrptw", n, m)
            ref = torch.load(dir_path + "ort_late{:02d}.pyth".format(late_p), map_location='cpu')["costs"]
            ort_optim = get_mean_std_gap(ref, ref)
            ort_expect = get_mean_std_gap(torch.load(dir_path + "ort_expected{:02d}_sampled.pyth".format(late_p),
                map_location='cpu'), ref)

            mardam_g = get_mean_std_gap(torch.load(dir_path + "mardan_late{:02d}.pyth".format(late_p),
                map_location='cpu'), ref)

            _, best = min((res[2],k) for k, res in enumerate((ort_optim, ort_expect, mardam_g)))

            tab.append([REF_CELL_STR.format(r"\boldsymbol" if best==0 else "", *ort_optim),
                CELL_STR.format(r"\boldsymbol" if best==1 else "", *ort_expect),
                CELL_STR.format(r"\boldsymbol" if best==2 else "", *mardam_g)])

        tab = [row for row in zip(*tab)]
        print('\n'.join(r"&{} &{} &{} &{} \\".format(mtd.ljust(23), *row)
            for mtd, row in zip(("ORTools (optimistic)", "ORTools (expected val.)", "MARDAM (g)"), tab)))
        print(r"\hline")

    print(r"\end{tabular}")
    print()

    ############ DYN
    print(r"""\begin{tabular}{|c|c|c c|c c|}
\hline
$r_\text{dyn}$ &\diagbox{Method}{dim.} &\multicolumns{2}{c|}{$N = 10$} &\mutlicolumn{2}{c|}{$N = 20$} \\
\hline
& &(Cost) &(QoS) &(Cost) &(QoS) \\
\multirow{2}{*}{0\%} %""")
    tab = []
    for n in (10, 20):
        m = n // 5
        dir_path = "./results/{}_n{}m{}/".format("sd_cvrptw", n, m)
        ref = torch.load(dir_path + "ort_oracle.pyth", map_location='cpu')["costs"]
        ort = get_mean_std_gap(ref, ref)
        mardam = torch.load(dir_path + "mardan_greedy.pyth", map_location='cpu')
        mardam_g = get_mean_std_gap(mardam["costs"], ref)

        _, best = min((res[2],k) for k, res in enumerate((ort, mardam_g)))

        tab.append([REF_CELL_STR.format(r"\boldsymbol" if best==0 else "", *ort),
            CELL_STR.format(r"\boldsymbol" if best==1 else "", *mardam_g)])

    tab = [row for row in zip(*tab)]
    print('\n'.join(r"&{} &{} &{} &{} \\".format(mtd.ljust(23), *row)
        for mtd, row in zip(("ORTools", "MARDAM (g)"), tab)))
    print(r"\hline")

    for dod in ("leq40", "less60", "geq60"):
        print(r"\multirow{{2}}{{*}}{{{}\%}} %".format(
            {"leq40": r"\leq 40", "less60": r"< 60", "geq60": r"\geq 60"}[dod]))
        tab = []
        for n in (10, 20):
            m = n // 5
            dir_path = "./results/{}_n{}m{}/".format("sd_cvrptw", n, m)
            ref = torch.load(dir_path + "best_insert_{}.pyth".format(dod), map_location='cpu')
            bi = ref["costs"].mean(), ref["costs"].std(), torch.tensor(0)
            bi_qos = ref["qos"].mean()
            mardam = torch.load(dir_path + "mardan_{}.pyth".format(dod), map_location='cpu')
            mardam_g = mardam["costs"].mean(), mardam["costs"].std(), (mardam["costs"] / ref["costs"] - 1).mean()
            mardam_qos = mardam["qos"].mean()

            best = mardam_g[2] <= 0
            best_qos = mardam_qos >= bi_qos

            tab.append([REF_CELL_STR.format("" if best else r"\boldsymbol", *bi),
                r"${}{{{:5.2f}\%}}$".format("" if best_qos else r"\boldsymbol", bi_qos),
                CELL_STR.format(r"\boldsymbol" if best else "", *mardam_g),
                r"${}{{{:5.2f}\%}}$".format(r"\boldsymbol" if best_qos else "", mardam_qos)])

        tab = [row for row in zip(*tab)]
        print('\n'.join(r"&{} &{} &{} &{} &{} \\".format(mtd.ljust(23), *row)
            for mtd, row in zip(("Best Insert", "MARDAM (g)"), tab)))
        print(r"\hline")

    print(r"\end{tabular}")
