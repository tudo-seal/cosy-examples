"""The algebras that read a synthesized term as a description of itself.

Each of them returns a dict from the name of a combinator to what that combinator stands for,
which is the form ``Tree.interpret`` reads.  A term is read here as a string for a person, as the
edge list of the computation graph it denotes, as a coarser term, as a request type another
synthesis can start from, and as a histogram over the operators it uses.  The algebras sit beside
the CNN repository rather than inside it, and none of them takes it as an argument.

None of these readings builds a network, so this module names no torch symbol and imports none.
The modules a combinator stands for, and the algebras that assemble them, live in
``cnn_damg_network_algebras``.  Nothing here imports from there, and nothing there imports from
here.
"""

import numpy as np
from cosy.core.tree import Tree
from cosy.core.types import Constructor, Literal


# Read a term as the string a person reads.  A leaf becomes its label with the input and output
# width it was synthesized at, a parallel composition becomes ``x || y``, a sequential one becomes
# ``x ; y``, and the learner becomes a block naming its model, loss, optimizer, schedule and epoch
# count.  A term folded to a coarser granularity keeps the same shape, which is why the clause
# names ending in ``_h1``, ``_h2`` and ``_h3`` read the same way as the ones they replace.
def pretty_term_algebra():
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: f"edges({io})"),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: f"swap({io}, {n}, {m})"),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({l!s}, {i}, {o})"),

            "conv2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({l!s}, {i}, {o})"),

            "maxpool2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({l!s}, {i}, {o})"),

            "batchnorm2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({l!s}, {i}, {o})"),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({l!s}, {i}, {o})"),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({l!s}, {i}, {o})"),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({l!s}, {i}, {o})"),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({l!s}, {i}, {o})"),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({l!s}, {i}, {o})"),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({l!s}, {i}, {o})"),

            "beside_singleton": (lambda i, o, ls, para, x: f"{x})"),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: f"{x} || {y}"),

            "before_singleton": (lambda i, o, r, ls, ls1, x: f"({x}"),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: f"({x} ; {y}"),

            "mse_loss": (lambda l: str(l)),

            "l1loss": (lambda l: str(l)),

            "cross_entropy_loss": (lambda l: str(l)),

            "adam_optimizer": (lambda o: str(o)),

            "sgd_optimizer": (lambda o: str(o)),

            "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer, str(sched))),

            "no_scheduler": (lambda opti, sched, optimizer: (optimizer, str(sched))),

            "learner": (lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: f"""
Learner(
    model= (
        {model}
        ), 
    loss= {loss}, 
    optimizer= {scheduler[0]}, 
    scheduler= {scheduler[1]}, 
    epochs= {e}
    )
"""),
            "edges_h1": (lambda io: f"edges({io})"),

            "swap_h1": (
                lambda io, n, m: f"swap({io}, {n}, {m})"),

            "linear_layer_h1": (lambda l, i, o: f"({l!s}, {i}, {o})"),

            "conv2d_h1": (lambda l, i, o: f"({l!s}, {i}, {o})"),

            "maxpool2d_h1": (lambda l, i, o: f"({l!s}, {i}, {o})"),

            "batchnorm2d_h1": (lambda l, i, o: f"({l!s}, {i}, {o})"),

            "sigmoid_h1": (lambda l, i, o: f"({l!s}, {i}, {o})"),

            "relu_h1": (lambda l, i, o: f"({l!s}, {i}, {o})"),

            "tanh_h1": (lambda l, i, o: f"({l!s}, {i}, {o})"),

            "sum_h1": (lambda l, i, o: f"({l!s}, {i}, {o})"),

            "product_h1": (lambda l, i, o: f"({l!s}, {i}, {o})"),

            "copy_h1": (lambda l, i, o: f"({l!s}, {i}, {o})"),

            "beside_singleton_h1": (lambda i, o, x: f"{x})"),

            "beside_cons_h1": (lambda i, i1, i2, o, o1, o2, x, y: f"{x} || {y}"),

            "before_singleton_h1": (lambda i, o, x: f"({x}"),

            "before_cons_h1": (lambda i, j, o, x, y: f"({x} ; {y}"),

            "learner_h1": (lambda i, o, e, loss, optimizer, scheduler, model: f"""
        Learner(
            model= (
                {model}
                ), 
            loss= {loss}, 
            optimizer= {optimizer}, 
            scheduler= {scheduler}, 
            epochs= {e}
            )
        """),
            "node": lambda i, o: "node",

            "loss": "loss",

            "optimizer": "optimizer",

            "scheduler": "scheduler",

            "scheduler_h1": (lambda s: f"{s!s}"),

            #"beside_singleton_h2": (lambda i, x: f"{x})"),

            #"beside_cons_h2": (lambda i1, x, y: f"{x} || {y}"),

            #"before_singleton_h2": (lambda i, x: f"({x}"),

            #"before_cons_h2": (lambda i, x, y: f"({x} ; {y}"),

            "learner_h2": (lambda e, loss, optimizer, scheduler, model: f"""
                Learner(
                    model= (
                        {model}
                        ), 
                    loss= {loss}, 
                    optimizer= {optimizer}, 
                    scheduler= {scheduler}, 
                    epochs= {e}
                    )
                """),

            "beside_singleton_h3": (lambda i, o, x: f"{x})"),

            "beside_cons_h3": (lambda i, i1, o, x, y: f"{x} || {y}"),
        }

def edgelist_learner(model, loss, optimizer, epochs, scheduler=None, verbose=False):
        # The tail of the chain is the model, then the loss, then the optimizer, then the
        # schedule, then the epoch count.  The schedule gets a node of its own because the graph
        # is what a graph kernel reads, so leaving it out would make two terms that differ only in
        # their schedule the same graph.  A tree built without one passes ``scheduler=None`` and
        # gets the shorter tail.
        f, inputs = model
        edgelist, to_outputs, pos_A = f((-5.5, -3.8), ["input" for _ in range(inputs)])
        tail = [loss, optimizer] if scheduler is None else [loss, optimizer, scheduler]
        tail = tail + [f"epochs({epochs})"]
        edgelist = edgelist + [(o, loss) for o in to_outputs]
        edgelist = edgelist + [(tail[k], tail[k + 1]) for k in range(len(tail) - 1)]
        if verbose:
            output_x = max([x for x, y in pos_A.values()]) + 2.5
            pos_A = pos_A | {"input": (-5.5, -3.8), "output": (output_x, -3.8)}
            pos_A = pos_A | {node: (output_x + 2.5 * (k + 1), -3.8) for k, node in enumerate(tail[:-1])}
            pos_A = pos_A | {tail[-1]: (output_x + 2.5 * len(tail[:-1]), -3.5)}
            return edgelist, pos_A
        return edgelist

# Read a term as the edge list of the computation graph it denotes, one edge per pair of node
# names.  A node is named after the label of the combinator that placed it, so handing the list to
# a graph constructor such as ``networkx.DiGraph`` builds the directed acyclic multigraph the term
# encodes, which is the computation graph of the network.
#
# Parallel and sequential composition of two edge-list continuations.
#
# These two helpers exist so that each continuation is evaluated exactly once, which is what makes
# the interpretation of a term linear in its size.  Written out inline as (x(...)[0] + y(...)[0],
# x(...)[1] + y(...)[1], x(...)[2] | y(...)[2]), a composition calls x and y once per component of
# the tuple it builds, and since the continuations are themselves built from these same
# combinators, that multiplies through the nesting instead of adding up.
#
# Reusing the value is not an approximation.  A continuation is pure, so one evaluation yields the
# same edges, the same outputs and the same node positions as three.
def _beside_edgelists(x, y, i1, id, inputs):
    """Parallel composition: x takes the first i1 inputs, y the rest."""
    left = x(id, inputs[:i1])
    right = y((id[0], id[1] + 0.2), inputs[i1:])
    return left[0] + right[0], left[1] + right[1], left[2] | right[2]


def _before_edgelists(x, y, id, inputs):
    """Sequential composition: y consumes the outputs of x.  y is a (continuation, arity) pair."""
    first = x(id, inputs)
    second = y[0]((id[0] + 2.5, id[1]), first[1])
    return second[0] + first[0], second[1], second[2] | first[2]


def edgelist_algebra(verbose=False, deduplicate=False):
        """Interpret a term as the edge list of the computation graph it denotes.

        Args:
            verbose (bool): Also return the map from node name to layout position, which a caller
                needs in order to strip the position back out of the name. (Default value = False)
            deduplicate (bool): Emit one edge per distinct source instead of one per input
                feature. (Default value = False)

        ``deduplicate`` is off by default because the multiplicity is not noise.  It is the width
        of the tensor: a layer with n input features receives n values, and an edge list that says
        so is the honest model of the data flow.

        The flag is there for a consumer that cannot read that width.  A graph kernel reading an
        adjacency structure sees a single entry for a pair of nodes however many parallel edges
        join them, so for that consumer the width is built and then discarded, and building it is
        what the conversion costs.  Which of the two a caller wants is therefore written at the
        call site instead of being decided here.
        """
        # ``dict.fromkeys`` rather than ``set``: the edge order follows the input order, so a
        # deduplicated list is a prefix-preserving thinning of the full one rather than a
        # reshuffle, which keeps the two comparable edge for edge.
        sources = dict.fromkeys if deduplicate else (lambda inputs: inputs)
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: ([], inputs, {})),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: ([], inputs[n:] + inputs[:n], {})),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "conv2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "maxpool2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "batchnorm2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "beside_singleton": (lambda i, o, ls, para, x: x),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: lambda id, inputs:
                _beside_edgelists(x, y, i1, id, inputs)),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x, i)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (
                lambda id, inputs: _before_edgelists(x, y, id, inputs), i)),

            "mse_loss": (lambda l: str(l)),

            "l1loss": (lambda l: str(l)),

            "cross_entropy_loss": (lambda l: str(l)),

            "adam_optimizer": (lambda o: str(o)),

            "sgd_optimizer": (lambda o: str(o)),

            "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer, sched)),

            "no_scheduler": (lambda opti, sched, optimizer: (optimizer, sched)),

            "learner": (lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: edgelist_learner(model, loss, scheduler[0], e, scheduler=scheduler[1], verbose=verbose)),

            "edges_h1": (lambda io: lambda id, inputs: ([], inputs, {})),

            "swap_h1": (
                lambda io, n, m: lambda id, inputs: ([], inputs[n:] + inputs[:n], {})),

            "linear_layer_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "conv2d_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "maxpool2d_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "batchnorm2d_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "sigmoid_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "relu_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "tanh_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "sum_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "product_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "copy_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in sources(inputs)], [str((l, i, o)) + str(id) for _ in range(o)],
                {str((l, i, o)) + str(id): id})),

            "beside_singleton_h1": (lambda i, o, x: x),

            "beside_cons_h1": (lambda i, i1, i2, o, o1, o2, x, y: lambda id, inputs:
                _beside_edgelists(x, y, i1, id, inputs)),

            "before_singleton_h1": (lambda i, o, x: (x, i)),

            "before_cons_h1": (lambda i, j, o, x, y: (
                lambda id, inputs: _before_edgelists(x, y, id, inputs), i)),

            "learner_h1": (lambda i, o, e, loss, optimizer, scheduler, model: edgelist_learner(model, loss, optimizer, e, scheduler=scheduler, verbose=verbose)),

            "node": (lambda i, o: lambda id, inputs: (
                [(x, "node" + str(id)) for x in sources(inputs)], ["node" + str(id) for _ in range(o)],
                {"node" + str(id): id})),

            "loss": "loss",

            "optimizer": "optimizer",

            "scheduler": "scheduler",

            "scheduler_h1": (lambda s: f"{s!s}"),

            #"beside_singleton_h2": (lambda i, x: x),

            #"beside_cons_h2": (lambda i1, x, y: lambda id, inputs:
            #    (x(id, inputs[:i1])[0] + y((id[0], id[1] + 0.2), inputs[i1:])[0],
            #     x(id, inputs[:i1])[1] + y((id[0], id[1] + 0.2), inputs[i1:])[1],
            #     x(id, inputs[:i1])[2] | y((id[0], id[1] + 0.2), inputs[i1:])[2])),

            #"before_singleton_h2": (lambda i, x: (x, i)),

            #"before_cons_h2": (lambda i, x, y: (lambda id, inputs:
            #                                                          (
            #                                                               y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[0] + x(id, inputs)[0],
            #                                                               y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[1],
            #                                                               y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[2] | x(id, inputs)[2]
            #                                                           ),
            #                                                           i)),

            "learner_h2": (lambda e, loss, optimizer, scheduler, model: edgelist_learner(model, loss, optimizer, e, scheduler=scheduler, verbose=verbose)),

            "beside_singleton_h3": (lambda i, o, x: x),

            "beside_cons_h3": (lambda i, i1, o, x, y: lambda id, inputs:
                _beside_edgelists(x, y, i1, id, inputs)),
        }


def hierarchy_algebra(level: int):
    # Levels 0, 1, 2 and 3 exist.  Any other level is read as level 0.
    if level == 1:
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                             para13, para14, para15, para16: Tree("edges_h1", (Tree(io),))),

            "swap": (
                lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                       para13, para14, para15, para16: Tree("swap_h1",  (Tree(io), Tree(n), Tree(m),))),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("linear_layer_h1", (Tree(l), Tree(i), Tree(o),))),

            "conv2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("conv2d_h1", (Tree(l), Tree(i), Tree(o),))),

            "maxpool2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("maxpool2d_h1", (Tree(l), Tree(i), Tree(o),))),

            "batchnorm2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("batchnorm2d_h1", (Tree(l), Tree(i), Tree(o),))),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("sigmoid_h1", (Tree(l), Tree(i), Tree(o),))),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("relu_h1", (Tree(l), Tree(i), Tree(o),))),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("tanh_h1", (Tree(l), Tree(i), Tree(o),))),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("sum_h1", (Tree(l), Tree(i), Tree(o),))),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("product_h1", (Tree(l), Tree(i), Tree(o),))),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("copy_h1", (Tree(l), Tree(i), Tree(o),))),

            "beside_singleton": (lambda i, o, ls, para, x: Tree("beside_singleton_h1", (Tree(i), Tree(o), x,))),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: Tree("beside_cons_h1", (Tree(i), Tree(i1), Tree(i2), Tree(o), Tree(o1), Tree(o2), x, y,))),

            "before_singleton": (lambda i, o, r, ls, ls1, x: Tree("before_singleton_h1",(Tree(i), Tree(o), x,))),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: Tree("before_cons_h1", (Tree(i), Tree(j), Tree(o), x, y,))),

             "mse_loss": (lambda l: Tree("mse_loss", (Tree(l),))),

            "l1loss": (lambda l: Tree("l1loss", (Tree(l),))),

            "cross_entropy_loss": (lambda l: Tree("cross_entropy_loss", (Tree(l),))),

            "adam_optimizer": (lambda o: Tree("adam_optimizer", (Tree(o),))),

            "sgd_optimizer": (lambda o: Tree("sgd_optimizer", (Tree(o),))),

            "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer, Tree("scheduler_h1", (Tree(sched),)))),

            "no_scheduler": (lambda opti, sched, optimizer: (optimizer, Tree("scheduler_h1", (Tree(sched),)))),

            "learner": (lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: Tree("learner_h1",
                                                                                    (Tree(i), Tree(o), Tree(e), loss, scheduler[0], scheduler[1], model,)))
        }
    if level == 2:
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                             para13, para14, para15, para16: Tree("edges_h1", (Tree(io),))),

            "swap": (
                lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                       para13, para14, para15, para16: Tree("swap_h1",  (Tree(io), Tree(n), Tree(m),))),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),
            "conv2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "maxpool2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "batchnorm2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),


            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "beside_singleton": (lambda i, o, ls, para, x: Tree("beside_singleton_h1", (Tree(i), Tree(o), x,))),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: Tree("beside_cons_h1", (Tree(i), Tree(i1), Tree(i2), Tree(o), Tree(o1), Tree(o2), x, y,))),

            "before_singleton": (lambda i, o, r, ls, ls1, x: Tree("before_singleton_h1",(Tree(i), Tree(o), x,))),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: Tree("before_cons_h1", (Tree(i), Tree(j), Tree(o), x, y,))),

            "mse_loss": (lambda l: Tree("loss")),

            "l1loss": (lambda l: Tree("loss")),
            "cross_entropy_loss": (lambda l: Tree("loss")),


            "adam_optimizer": (lambda o: Tree("optimizer")),

            "sgd_optimizer": (lambda o: Tree("optimizer")),

            "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer, Tree("scheduler"))),

            "no_scheduler": (lambda opti, sched, optimizer: (optimizer, Tree("scheduler"))),

            "learner": (lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: Tree("learner_h2",
                                                                                    (Tree(e), loss, scheduler[0], scheduler[1], model,)))
        }
    elif level == 3:
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                             para13, para14, para15, para16: Tree("edges_h1", (Tree(io),))),

            "swap": (
                lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                       para13, para14, para15, para16: Tree("swap_h1",  (Tree(io), Tree(n), Tree(m),))),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),
            "conv2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "maxpool2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "batchnorm2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),


            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "beside_singleton": (lambda i, o, ls, para, x: Tree("beside_singleton_h3", (Tree(i), Tree(o), x,))),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y:
                            Tree("beside_cons_h3", (Tree(i), Tree(i1), Tree(o), x, y,))
                            if x.root == "edges_h1" or x.root == "swap_h1"
                            else (Tree("beside_singleton_h3", (Tree(i), Tree(o), y.children[2]))
                                  if y.root == "beside_singleton_h3"
                                  else Tree("beside_cons_h3",
                                            (Tree(i), Tree(i1), Tree(o), Tree("node", (Tree(i1), Tree(o1),)), y.children[4])))),

            "before_singleton": (lambda i, o, r, ls, ls1, x: Tree("before_singleton_h1",(Tree(i), Tree(o), x,))),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: Tree("before_cons_h1", (Tree(i), Tree(j), Tree(o), x, y,))),

            "mse_loss": (lambda l: Tree("loss")),

            "l1loss": (lambda l: Tree("loss")),
            "cross_entropy_loss": (lambda l: Tree("loss")),


            "adam_optimizer": (lambda o: Tree("optimizer")),

            "sgd_optimizer": (lambda o: Tree("optimizer")),

            "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer, Tree("scheduler"))),

            "no_scheduler": (lambda opti, sched, optimizer: (optimizer, Tree("scheduler"))),

            "learner": (lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: Tree("learner_h2",
                                                                                    (Tree(e), loss, scheduler[0], scheduler[1], model,)))
        }
    else:
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                             para13, para14, para15, para16: Tree("edges",
                                                                  (
                                                                      Tree(io), Tree(para1), Tree(para2), Tree(para3),
                                                                      Tree(para4), Tree(para5), Tree(para6),
                                                                      Tree(para7),
                                                                      Tree(para8), Tree(para9), Tree(para10),
                                                                      Tree(para11), Tree(para12), Tree(para13),
                                                                      Tree(para14), Tree(para15), Tree(para16),
                                                                  ))),

            "swap": (
                lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                       para13, para14, para15, para16: Tree("swap",
                                                            (
                                                                Tree(io), Tree(n), Tree(m),
                                                                Tree(para1), Tree(para2), Tree(para3),
                                                                Tree(para4), Tree(para5), Tree(para6),
                                                                Tree(para7),
                                                                Tree(para8), Tree(para9), Tree(para10),
                                                                Tree(para11), Tree(para12), Tree(para13),
                                                                Tree(para14), Tree(para15), Tree(para16),
                                                            ))),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("linear_layer",
                                                                                                   (
                                                                                                       Tree(l), Tree(i),
                                                                                                       Tree(o),
                                                                                                       Tree(para1),
                                                                                                       Tree(para2),
                                                                                                       Tree(para3),
                                                                                                       Tree(para4),
                                                                                                       Tree(para5),
                                                                                                       Tree(para6),
                                                                                                       Tree(para7),
                                                                                                   ))),

            "conv2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("conv2d",
                                                                                                   (
                                                                                                       Tree(l), Tree(i),
                                                                                                       Tree(o),
                                                                                                       Tree(para1),
                                                                                                       Tree(para2),
                                                                                                       Tree(para3),
                                                                                                       Tree(para4),
                                                                                                       Tree(para5),
                                                                                                       Tree(para6),
                                                                                                       Tree(para7),
                                                                                                   ))),

            "maxpool2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("maxpool2d",
                                                                                                   (
                                                                                                       Tree(l), Tree(i),
                                                                                                       Tree(o),
                                                                                                       Tree(para1),
                                                                                                       Tree(para2),
                                                                                                       Tree(para3),
                                                                                                       Tree(para4),
                                                                                                       Tree(para5),
                                                                                                       Tree(para6),
                                                                                                       Tree(para7),
                                                                                                   ))),

            "batchnorm2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("batchnorm2d",
                                                                                                   (
                                                                                                       Tree(l), Tree(i),
                                                                                                       Tree(o),
                                                                                                       Tree(para1),
                                                                                                       Tree(para2),
                                                                                                       Tree(para3),
                                                                                                       Tree(para4),
                                                                                                       Tree(para5),
                                                                                                       Tree(para6),
                                                                                                       Tree(para7),
                                                                                                   ))),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("sigmoid",
                                                                                              (
                                                                                                  Tree(l), Tree(i),
                                                                                                  Tree(o),
                                                                                                  Tree(para1),
                                                                                                  Tree(para2),
                                                                                                  Tree(para3),
                                                                                                  Tree(para4),
                                                                                                  Tree(para5),
                                                                                                  Tree(para6),
                                                                                                  Tree(para7),
                                                                                              ))),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("relu",
                                                                                           (
                                                                                               Tree(l), Tree(i),
                                                                                               Tree(o),
                                                                                               Tree(para1), Tree(para2),
                                                                                               Tree(para3),
                                                                                               Tree(para4), Tree(para5),
                                                                                               Tree(para6),
                                                                                               Tree(para7),
                                                                                           ))),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("tanh",
                                                                                           (
                                                                                               Tree(l), Tree(i),
                                                                                               Tree(o),
                                                                                               Tree(para1), Tree(para2),
                                                                                               Tree(para3),
                                                                                               Tree(para4), Tree(para5),
                                                                                               Tree(para6),
                                                                                               Tree(para7),
                                                                                           ))),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("sum",
                                                                                          (
                                                                                              Tree(l), Tree(i), Tree(o),
                                                                                              Tree(para1), Tree(para2),
                                                                                              Tree(para3),
                                                                                              Tree(para4), Tree(para5),
                                                                                              Tree(para6),
                                                                                              Tree(para7),
                                                                                          ))),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("product",
                                                                                              (
                                                                                                  Tree(l), Tree(i),
                                                                                                  Tree(o),
                                                                                                  Tree(para1),
                                                                                                  Tree(para2),
                                                                                                  Tree(para3),
                                                                                                  Tree(para4),
                                                                                                  Tree(para5),
                                                                                                  Tree(para6),
                                                                                                  Tree(para7),
                                                                                              ))),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("copy",
                                                                                           (
                                                                                               Tree(l), Tree(i),
                                                                                               Tree(o),
                                                                                               Tree(para1), Tree(para2),
                                                                                               Tree(para3),
                                                                                               Tree(para4), Tree(para5),
                                                                                               Tree(para6),
                                                                                               Tree(para7),
                                                                                           ))),

            "beside_singleton": (lambda i, o, ls, para, x: Tree("beside_singleton",
                                                                (
                                                                    Tree(i), Tree(o),
                                                                    Tree(ls), Tree(para), x
                                                                ))),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: Tree("beside_cons",
                                                                                    (
                                                                                        Tree(i), Tree(i1), Tree(i2),
                                                                                        Tree(o), Tree(o1), Tree(o2),
                                                                                        Tree(ls), Tree(head),
                                                                                        Tree(tail),
                                                                                        x, y,
                                                                                    ))),

            "before_singleton": (lambda i, o, r, ls, ls1, x: Tree("before_singleton",
                                                                  (
                                                                      Tree(i), Tree(o), Tree(r),
                                                                      Tree(ls), Tree(ls1), x,
                                                                  ))),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: Tree("before_cons",
                                                                          (
                                                                              Tree(i), Tree(j), Tree(o),
                                                                              Tree(r), Tree(ls), Tree(head), Tree(tail),
                                                                              x, y,
                                                                          ))),

            "mse_loss": (lambda l: Tree("mse_loss", (Tree(l),))),

            "l1loss": (lambda l: Tree("l1loss", (Tree(l),))),
            "cross_entropy_loss": (lambda l: Tree("cross_entropy_loss", (Tree(l),))),


            "adam_optimizer": (lambda o: Tree("adam_optimizer", (Tree(o),))),

            "sgd_optimizer": (lambda o: Tree("sgd_optimizer", (Tree(o),))),

            "cosine_annealing_lr": (lambda opti, sched, optimizer: Tree("cosine_annealing_lr", (Tree(opti), Tree(sched), optimizer))),

            "no_scheduler": (lambda opti, sched, optimizer: Tree("no_scheduler", (Tree(opti), Tree(sched), optimizer))),

            "learner": (lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: Tree("learner",
                                                                                    (
                                                                                        Tree(i), Tree(o), Tree(r),
                                                                                        Tree(ls), Tree(e), Tree(l),
                                                                                        Tree(opt), Tree(sch),
                                                                                        loss, scheduler, model,
                                                                                    )))
        }

# Read a term as the request type that names it.  Every leaf keeps its label and both of its
# widths, which makes this the most specific of the three requests in this module.  The two
# refinements below open part of it.
def request_algebra():
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                             para13, para14, para15, para16: (("swap", 0, io), io, io)),

            "swap": (
                lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                       para13, para14, para15, para16: (("swap", n, m), io, io)),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),
            "conv2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "maxpool2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "batchnorm2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),


            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "beside_singleton": (lambda i, o, ls, para, x: (x,)),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: (x, *y)),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x,)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (x, *y)),

            "mse_loss": (lambda l: Literal(l)),

            "l1loss": (lambda l: Literal(l)),
            "cross_entropy_loss": (lambda l: Literal(l)),


            "adam_optimizer": (lambda o: Literal(o)),

            "sgd_optimizer": (lambda o: Literal(o)),

            "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer, Literal(sched))),

            "no_scheduler": (lambda opti, sched, optimizer: (optimizer, Literal(sched))),

            "learner": (lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: Constructor("Learner", Constructor("DAG",
                          Constructor("input", Literal(i))
                          & Constructor("output", Literal(o))
                          & Constructor("structure", Literal(model)))
                                & Constructor("Loss", Constructor("type", loss))
                                & Constructor("Optimizer", Constructor("type", scheduler[0]))
                                & Constructor("Scheduler", Constructor("type", scheduler[1]))
                                & Constructor("epochs", Literal(e))
                                ))
        }


# Open every leaf and keep only the shape of the term.  What a request built from this fixes is
# the nesting of the compositions and the two widths the learner carries, not what stands in any
# one position.
def refinement_1_algebra():
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: None),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: None),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),
            "conv2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "maxpool2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "batchnorm2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),


            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "beside_singleton": (lambda i, o, ls, para, x: (x,)),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: (x, *y)),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x,)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (x, *y)),

            "mse_loss": (lambda l: Literal(None)),

            "l1loss": (lambda l: Literal(None)),
            "cross_entropy_loss": (lambda l: Literal(None)),


            "adam_optimizer": (lambda o: Literal(None)),

            "sgd_optimizer": (lambda o: Literal(None)),

            "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer, Literal(None))),

            "no_scheduler": (lambda opti, sched, optimizer: (optimizer, Literal(None))),

            "learner": (lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: Constructor("Learner", Constructor("DAG",
                          Constructor("input", Literal(i))
                          & Constructor("output", Literal(o))
                          & Constructor("structure", Literal(model)))
                                & Constructor("Loss", Constructor("type", loss))
                                & Constructor("Optimizer", Constructor("type", scheduler[0]))
                                & Constructor("Scheduler", Constructor("type", scheduler[1]))
                                & Constructor("epochs", Literal(e))
                                ))
        }


# Open the label of every leaf and keep the input and output width it was synthesized at.  A
# request built from this reaches the terms of the same shape carrying another label in each
# position.  ``refinement_1_algebra`` above opens the widths along with the label, so a request
# built from that one reaches terms of other widths as well.
def refinement_2_algebra():
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: (("swap", 0, io), io, io)),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: (("swap", n, m), io, io)),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),
            "conv2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "maxpool2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "batchnorm2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),


            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "beside_singleton": (lambda i, o, ls, para, x: (x,)),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: (x, *y)),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x,)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (x, *y)),

            "mse_loss": (lambda l: Literal(None)),

            "l1loss": (lambda l: Literal(None)),
            "cross_entropy_loss": (lambda l: Literal(None)),


            "adam_optimizer": (lambda o: Literal(None)),

            "sgd_optimizer": (lambda o: Literal(None)),

            "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer, Literal(None))),

            "no_scheduler": (lambda opti, sched, optimizer: (optimizer, Literal(None))),

            "learner": (lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: Constructor("Learner", Constructor("DAG",
                          Constructor("input", Literal(i))
                          & Constructor("output", Literal(o))
                          & Constructor("structure", Literal(model)))
                                & Constructor("Loss", Constructor("type", loss))
                                & Constructor("Optimizer", Constructor("type", scheduler[0]))
                                & Constructor("Scheduler", Constructor("type", scheduler[1]))
                                & Constructor("epochs", Literal(e))
                                ))
        }


# Normalize a histogram to relative frequencies, in place.  A vector that counted nothing is left
# alone, since it has nothing to divide by.
def operator_histogram(vec):
    if vec.sum() > 0:
        vec /= vec.sum()

    return vec

#: The axes of the histogram, in order.  A new combinator adds itself here and nowhere else, and
#: ``_one_hot`` raises for an operator that is not an axis.  Append, never insert: the position is
#: the axis, so an insertion renumbers every operator after it and silently reinterprets every
#: histogram computed before the change.
_HISTOGRAM_OPERATORS = (
    "edges", "swap", "linear_layer", "conv2d", "maxpool2d", "sigmoid", "relu", "tanh", "sum",
    "product", "copy", "batchnorm2d",
)


def _one_hot(operator):
    """The unit vector of one operator over ``_HISTOGRAM_OPERATORS``.

    A fresh array per call, which matters: ``operator_histogram`` normalizes in place (``vec /=``),
    so a shared array would be mutated by the first term that used it.

    Args:
        operator (str): The combinator name, which must be an axis.

    Returns:
        numpy.ndarray: The one-hot vector, length ``len(_HISTOGRAM_OPERATORS)``.
    """
    vec = np.zeros(len(_HISTOGRAM_OPERATORS), dtype=float)
    vec[_HISTOGRAM_OPERATORS.index(operator)] = 1.0
    return vec


# Read a term as the histogram of the operators it uses, one axis per entry of
# ``_HISTOGRAM_OPERATORS``.  The learner clause normalizes the counts to relative frequencies, so
# two terms of different size are comparable.
def operator_histogram_algebra():
    return {
        "edges": (
            lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13,
                   para14, para15, para16: _one_hot("edges")),

        "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                        para13, para14, para15, para16: _one_hot("swap")),

        "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: _one_hot("linear_layer")),

        "conv2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: _one_hot("conv2d")),

        "maxpool2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: _one_hot("maxpool2d")),

        "batchnorm2d": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: _one_hot("batchnorm2d")),

        "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: _one_hot("sigmoid")),

        "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: _one_hot("relu")),

        "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: _one_hot("tanh")),

        "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: _one_hot("sum")),

        "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: _one_hot("product")),

        "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: _one_hot("copy")),

        "beside_singleton": (lambda i, o, ls, para, x: x),

        "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: x + y),

        "before_singleton": (lambda i, o, r, ls, ls1, x: x),

        "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: x + y),

        "mse_loss": (lambda l: ()),

        "l1loss": (lambda l: ()),

        "cross_entropy_loss": (lambda l: ()),

        "adam_optimizer": (lambda o: ()),

        "sgd_optimizer": (lambda o: ()),

        "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer, ())),

        "no_scheduler": (lambda opti, sched, optimizer: (optimizer, ())),

        "learner": (lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: operator_histogram(model))
    }
