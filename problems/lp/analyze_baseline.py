import numpy as np

fns = ["1ba.dimacs", "1bipartite.dimacs", "1bp_seed1234.dimacs",
       "1er.dimacs",
           "1path.dimacs", "1regular.dimacs"]

def parse_fn(fn):
    fn = 'results/' + fn.split('.')[0] + '.results'
    with open(fn) as f:
        lines = f.readlines()

    lengths = []
    for line in lines:
        _, a, b, _, c = line.strip().split()
        lengths.append((a, int(c)-1))
        lengths.append((b, int(c) - 1))

    opts = dict()
    for start, l in lengths:
        if start not in opts or l > opts[start]:
            opts[start] = l

    return fn, opts, np.mean(list(opts.values()))

def opt_routes(fn):
    fn = 'results/' + fn.split('.')[0] + '.routes'
    with open(fn) as f:
        lines = f.readlines()
    opts = dict()
    maxs = dict()
    for line in lines:
        route = line.strip().split(',')
        a = route[0]
        b = route[-1]
        if a not in opts or len(route) - 1 > maxs[a]:
            opts[a] = route
            maxs[a] = len(route) - 1

        if b not in opts or len(route) - 1 > maxs[b]:
            opts[b] = route
            maxs[b] = len(route) - 1

    return opts, maxs

for i in range(6):
    fn, opts, m = parse_fn(fns[i])
    print(fn, m)
# routes, maxs = opt_routes(fns[i])

console = []