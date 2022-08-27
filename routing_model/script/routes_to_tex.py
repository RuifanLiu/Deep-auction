from marpdan import AttentionLearner
from marpdan.problems import VRPTW_Dataset, VRPTW_Environment
from marpdan.externals import ort_solve
from marpdan.utils import eval_apriori_routes, load_old_weights

from itertools import cycle, tee
import torch
import subprocess
import os

N = 20
M = N // 5


TIKZ_TMPL = r"""\documentclass[tikz, crop]{{standalone}}
\usetikzlibrary{{shapes,positioning, backgrounds}}
\begin{{document}}
\begin{{tikzpicture}}[x=3cm, y=3cm,
    depot/.style = {{draw, fill=white, minimum height=5mm, diamond, aspect=0.7, font=\footnotesize}},
    cust/.style = {{draw, fill=white, font=\scriptsize}},
    tw/.style = {{font=\tiny, align=center}},
    every path/.append style = {{->, >=latex}}]

    \node[depot] (n0) at ({},{}) {{$0$}};
    {}

    \begin{{pgfonlayer}}{{background}}
        {}
    \end{{pgfonlayer}}
\end{{tikzpicture}}
\end{{document}}"""

COLORS = cycle(("blue", "green!50!black", "red", "orange"))

def prv_nxt(iterable):
    prv, nxt = tee(iterable, 2)
    yield (0,next(nxt))
    yield from zip(prv, nxt)


def tikz_node(node, idx):
    return r"""\node[cust, minimum width={3:.1f}mm, minimum height={3:.1f}mm] (n{0}) at ({1:.1f}, {2:.1f}) {{${0}$}};
    \node[tw, above = 0mm of n{0}] {{{4} -\\- {5}}};
    \node[tw, below = 0mm of n{0}] {{{6}min}};""".format(
        idx, node[0]/10, node[1]/10, node[2]/5+4,
        "{}:{:02d}".format(*divmod(int(node[3])+480, 60)), "{}:{:02d}".format(*divmod(int(node[4])+480, 60)),
        int(node[5]))


def tikz_route(route, col):
    return "\n        ".join(r"\draw[{}] (n{}) -- (n{});".format(col, prv, nxt) for prv,nxt in prv_nxt(route))


learner = AttentionLearner(6,4)
chkpt = torch.load("pretrained/cvrptw_n{}m{}.pyth".format(N,M), map_location='cpu')
load_old_weights(learner, chkpt["model"])
learner.eval()
learner.greedy = True

data = VRPTW_Dataset.generate(1, N, M, tw_ratio=1.0)
nodes = data.nodes[0].clone()

ort_routes = ort_solve(data)[0]

data.normalize()
env = VRPTW_Environment(data)
ort_cost = eval_apriori_routes(env, [ort_routes], 1)[0]
print("   ORT COST =", ort_cost.item())

actions, _, rewards = learner(env)
routes = [[] for _ in range(M)]
for i,j in actions:
    routes[i.item()].append(j.item())
print("MARDAM COST =", -torch.stack(rewards).sum(0).item())

with open("ortools_routes_n{}.tex".format(N), 'w') as f:
    print(TIKZ_TMPL.format(nodes[0,0]/10, nodes[0,1]/10,
        "\n    ".join(tikz_node(n, i) for i,n in enumerate(nodes[1:], start=1)),
        "\n\n        ".join(tikz_route(route, col) for route, col in zip(ort_routes, COLORS))), file=f)

with open("mardam_routes_n{}.tex".format(N), 'w') as f:
    print(TIKZ_TMPL.format(nodes[0,0]/10, nodes[0,1]/10,
        "\n    ".join(tikz_node(n, i) for i,n in enumerate(nodes[1:], start=1)),
        "\n\n        ".join(tikz_route(route, col) for route, col in zip(routes, COLORS))), file=f)

subprocess.run(["pdflatex", "-halt-on-error", "mardam_routes_n{}.tex".format(N)], stdout=subprocess.DEVNULL)
subprocess.run(["pdflatex", "-halt-on-error", "ortools_routes_n{}.tex".format(N)], stdout=subprocess.DEVNULL)
subprocess.run(["xreader", "mardam_routes_n{}.pdf".format(N), "ortools_routes_n{}.pdf".format(N)],
        stdout=subprocess.DEVNULL)
