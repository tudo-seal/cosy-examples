
from dataclasses import dataclass


from cosy.core.types import DataGroup, Constructor, Var, Literal, Group
from cosy.core import SpecificationBuilder, Synthesizer
from cosy.core.tree import Tree


VALID_REDUCTIONS = {"mean", "sum", "none"}


def _is_triplet_tuple(value) -> bool:
    # Common shape check for literal triples like (label, in_dim, out_dim).
    return isinstance(value, tuple) and len(value) == 3


def _is_singleton_tuple(value) -> bool:
    # Helper for sequential literals encoded as single-element tuples.
    return isinstance(value, tuple) and len(value) == 1


def _is_pair_tuple(value) -> bool:
    # Helper for pairwise rewrite patterns in tuple-tuples.
    return isinstance(value, tuple) and len(value) == 2


def _is_swap_label(value) -> bool:
    # Swap labels are encoded as ("swap", m, n).
    return _is_triplet_tuple(value) and value[0] == "swap"


def _swap_parts(value):
    # Extract (m, n) from a swap label ("swap", m, n), otherwise signal no-match.
    if not _is_swap_label(value):
        return None
    return value[1], value[2]


def _is_swap_zero(value) -> bool:
    # Edge-like swaps have m == 0.
    parts = _swap_parts(value)
    return parts is not None and parts[0] == 0


def _matches_swap_zero_pair(left, right) -> bool:
    # Match beside(swap(n,0,n), swap(m,0,m)) at literal level.
    if not (_is_triplet_tuple(left) and _is_triplet_tuple(right)):
        return False
    if left[0] is None or right[0] is None:
        return False
    return _is_swap_zero(left[0]) and _is_swap_zero(right[0])


def _tree_root_is(tree, root_name: str) -> bool:
    # Exact root comparison for Tree nodes.
    return tree.root == root_name


def _tree_root_contains(tree, token: str) -> bool:
    # Preserve the existing substring-based root matching used by the predicates.
    return token in tree.root


def _require_children_count(tree, expected: int):
    # Keep the shape checks and their error message in one place.
    if len(tree.children) != expected:
        raise ValueError("Derivation trees have not the expected shape.")
    return tree.children


def _tree_child(tree, index: int):
    # Short helper for indexing into a derivation tree.
    return tree.children[index]


class CNNrepository:
    """
            This repository models neural networks as directed acyclic graphs.
            The components/combinators of the repository are standard pytorch modules.
            The directed acyclic graph corresponds to the computational graph of a neural network.
            Or in the categorical model a string diagram.
            The combinators therefore correspond to string diagram constructors in a symmetric monoidal category.
            The algebraic theory of directed acyclic graphs as string diagrams for symmetric monoidal categories is
            inspired by Gibbons paper "An initial algebra approach to directed acyclic graphs".

            Following the idea of Gibbons paper, we have to ensure to synthesize the quotient of terms under the
            following rewrite rules:

            associativity laws:  (handled with types)

            beside(beside(x,y),z)
            ->
            beside(x, beside(y,z))

            before(before(x,y),z)
            ->
            before(x, before(y,z))

            abiding law:   (handled with types)

            beside(before(m,n,p, w(m,n), x(n,p)), before(m',r,p', y(m',r), z(r,p')))
            ->
            before(m+m', n+r, p+p', beside(w(m,n),y(m',r)), beside(x(n,p),z(r,p')))

            neutrality of edge:   (handled with types)

            before(edge(), x)
            ->
            x

            before(x, edge())
            ->
            x

            swap laws:    (they need to be term predicates :-(...)


            before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
            ->
            beside(y(m,q),x(n,p))


            before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
            ->
            swap(m + n + p, m, n+p)

            before(swap(m+n, m, n), swap(n+m, n, m))
            ->
            copy(m+n, edge())

            before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
            ->
            swap(m + n + p, m+n, p)


            additionally, we should satisfy the following law (which is implicit in Gibbon's paper...)
            (this is handled with types, too)

            beside(swap(n, 0, n), swap(m, 0, m))
            ->
            swap(n+m, 0, n+m)

            We can ensure that every synthesized term is in normalform under these rewrite rules by constructing
            specific intersection types and utilizing term predicates, where we can't encode a rule into the type.
    """

    def __init__(self,
                 linear_feature_dimensions: list[int], channel_dimensions: list[int],
                 height_width_dimensions: list[tuple[int, int]], kernel_dimensions: list[tuple[int, int]],
                 stride_values: list[int], padding_values: list[int],
                 constant_values: list[int], learning_rate_values: list[float],
                 n_epoch_values: list[int], dimensions=None):
        # TODO: What was the idea of the dimensions argument? Do we need it at all?
        self.dimensions: list[int] = list(range(1, max(linear_feature_dimensions) + 1) if (dimensions is None or
                                                                           (dimensions is not None and
                                                                            max(dimensions) <
                                                                            max(linear_feature_dimensions)))
                           else dimensions)
        self.channel_dimensions: list[int] = channel_dimensions
        self.higher_dimensions: list[tuple[int, int]] = height_width_dimensions
        self.kernel_dimensions: list[tuple[int, int]] = kernel_dimensions
        self.stride_values: list[int] = stride_values
        self.padding_values: list[int] = padding_values
        self.linear_feature_dimensions = linear_feature_dimensions
        self.learning_rate_values = learning_rate_values
        self.n_epoch_values = n_epoch_values
        self.constant_values = (constant_values if 1 in constant_values and 0 in constant_values
                                else constant_values + [0, 1])

    # We will interpret every combinator as a pytorch nn.module. We will treat combinators and their interpretations as
    # parametric functions. Our repository will therefore model the Para-construction on a symmetric monoidal category.
    # Parameters will be modeled as literals. Therefore, we need a dataclass for every component that describes the
    # possible parameters/literals for the synthesis.
    @dataclass(frozen=True)
    class Linear:
        in_features: int
        out_features: int
        bias: bool = True

    # We conservatively extend the DAMG-Repository with the usual constructions for convolutional neural networks.
    @dataclass(frozen=True)
    class Conv2d:
        in_channels: int
        out_channels: int
        input_size: tuple[int, int]
        output_size: tuple[int, int]
        kernel_size: tuple[int, int]
        stride: int = 1
        padding: int = 0
        bias: bool = True

    @dataclass(frozen=True)
    class MaxPool2d:
        in_channels: int
        out_channels: int
        input_size: tuple[int, int]
        output_size: tuple[int, int]
        kernel_size: tuple[int, int]
        stride: int = 1
        padding: int = 0

    @dataclass(frozen=True)
    class Flatten:
        in_channels: int
        input_size: tuple[int, int]
        out_features: int

    @dataclass(frozen=True)
    class Unflatten:
        in_features: int
        out_channels: int
        output_size: tuple[int, int]

    @dataclass(frozen=True)
    class Sigmoid:
        trivial = ()

    @dataclass(frozen=True)
    class ReLu:
        inplace: bool = False

    @dataclass(frozen=True)
    class Tanh:
        trivial = ()

    @dataclass(frozen=True)
    class Sum:
        with_constant: float = 0

    @dataclass(frozen=True)
    class Product:
        with_constant: float = 1

    @dataclass(frozen=True)
    class Copy:
        out_dimension: int

    # Labels are a dependent product of neural-network-component-names and their valid parameter tuples.
    class Label(Group):
        name = "Label"

        def __init__(self, dimensions,
                     linear_feature_dimensions, constant_values, channel_dimensions, kernel_dimensions,
                     stride_values, padding_values, higher_dimensions):
            self.dimensions = tuple(dimensions)
            self.dimension_set = set(self.dimensions)
            self.linear_feature_dimensions = tuple(linear_feature_dimensions)
            self.linear_feature_dimension_set = set(self.linear_feature_dimensions)
            self.constant_values = tuple(constant_values)
            self.constant_value_set = set(self.constant_values)
            self.channel_dimensions = tuple(channel_dimensions)
            self.kernel_dimensions = tuple(kernel_dimensions)
            self.stride_values = tuple(stride_values)
            self.padding_values = tuple(padding_values)
            self.higher_dimensions = tuple(higher_dimensions)
            self._iter_cache = None

        def iter_linear(self):
            for in_f in self.linear_feature_dimensions:
                for out_f in self.linear_feature_dimensions:
                    yield CNNrepository.Linear(in_features=in_f, out_features=out_f, bias=True)
                    yield CNNrepository.Linear(in_features=in_f, out_features=out_f, bias=False)

        def iter_conv2d(self):
            for in_c in self.channel_dimensions:
                for in_dim in self.higher_dimensions:
                        for k_dim in self.kernel_dimensions:
                            for stride in self.stride_values:
                                for padding in self.padding_values:
                                    out_dim: tuple[int, int] = (
                                        ((in_dim[0] + 2 * padding - (k_dim[0] - 1) - 1) // stride) + 1,
                                        ((in_dim[1] + 2 * padding - (k_dim[1] - 1) - 1) // stride) + 1,
                                    )
                                    if out_dim in self.higher_dimensions:
                                        yield CNNrepository.Conv2d(in_channels=in_c, out_channels=in_c, input_size=in_dim, output_size=out_dim, kernel_size=k_dim, stride=stride, padding=padding, bias=True)
                                        yield CNNrepository.Conv2d(in_channels=in_c, out_channels=in_c, input_size=in_dim, output_size=out_dim, kernel_size=k_dim, stride=stride, padding=padding, bias=False)

        def iter_maxpool2d(self):
            for in_c in self.channel_dimensions:
                for in_dim in self.higher_dimensions:
                    for k_dim in self.kernel_dimensions:
                        for stride in self.stride_values:
                            for padding in self.padding_values:
                                out_dim: tuple[int, int] = (
                                    ((in_dim[0] + 2 * padding - (k_dim[0] - 1) - 1) // stride) + 1,
                                    ((in_dim[1] + 2 * padding - (k_dim[1] - 1) - 1) // stride) + 1,
                                )
                                if out_dim in self.higher_dimensions:
                                    yield CNNrepository.MaxPool2d(in_channels=in_c, out_channels=in_c, input_size=in_dim, output_size=out_dim, kernel_size=k_dim, stride=stride, padding=padding)

        def iter_flatten(self):
            for in_c in self.channel_dimensions:
                for in_dim in self.higher_dimensions:
                    out_features = in_c * in_dim[0] * in_dim[1]
                    if out_features in self.linear_feature_dimensions:
                        yield CNNrepository.Flatten(in_channels=in_c, input_size=in_dim, out_features=out_features)

        def iter_unflatten(self):
            for out_c in self.channel_dimensions:
                for out_dim in self.higher_dimensions:
                    in_features = out_c * out_dim[0] * out_dim[1]
                    if in_features in self.linear_feature_dimensions:
                        yield CNNrepository.Unflatten(in_features=in_features, out_channels=out_c, output_size=out_dim)

        def iter_sigmoid(self):
            yield CNNrepository.Sigmoid()

        def iter_relu(self):
            yield CNNrepository.ReLu(inplace=False)
            #yield ODErepository.ReLu(inplace=True)  # inplace not supported here

        def iter_tanh(self):
            yield CNNrepository.Tanh()

        def iter_sum(self):
            for c in self.constant_values:
                yield CNNrepository.Sum(c)

        def iter_product(self):
            for c in self.constant_values:
                if c != 0:
                    yield CNNrepository.Product(c)

        def iter_copy(self):
            for o in self.dimensions:
                # Copy 1 would be the same as an edge, and we don't want redundancies!
                if o >= 2:
                    yield CNNrepository.Copy(out_dimension=o)

        def __iter__(self):
            if self._iter_cache is None:
                self._iter_cache = tuple(
                    [*self.iter_linear(), *self.iter_sigmoid(), *self.iter_relu(), *self.iter_tanh(),
                     *self.iter_sum(), *self.iter_product(), *self.iter_copy(),
                     *self.iter_conv2d(), *self.iter_maxpool2d(), *self.iter_flatten(), *self.iter_unflatten()]
                )
            yield from self._iter_cache

        def __contains__(self, item):
            if isinstance(item, CNNrepository.Linear):
                return ((item.in_features in self.linear_feature_dimension_set) and
                        (item.out_features in self.linear_feature_dimension_set) and
                        (isinstance(item.bias, bool)))
            elif isinstance(item, CNNrepository.Sigmoid):
                return True
            elif isinstance(item, CNNrepository.ReLu):
                return isinstance(item.inplace, bool)
            elif isinstance(item, CNNrepository.Tanh):
                return True
            elif isinstance(item, CNNrepository.Sum):
                return item.with_constant in self.constant_value_set
            elif isinstance(item, CNNrepository.Product):
                return item.with_constant in self.constant_value_set and item.with_constant != 0
            elif isinstance(item, CNNrepository.Copy):
                return item.out_dimension in self.dimension_set
            elif isinstance(item, CNNrepository.Conv2d):
                return ((item.in_channels in self.channel_dimensions) and
                        (item.out_channels in self.channel_dimensions) and
                        (item.input_size in self.higher_dimensions) and
                        (item.output_size in self.higher_dimensions) and
                        (item.kernel_size in self.kernel_dimensions) and
                        (item.stride in self.stride_values) and
                        (item.padding in self.padding_values) and
                        (isinstance(item.bias, bool)))
            elif isinstance(item, CNNrepository.MaxPool2d):
                return ((item.kernel_size in self.kernel_dimensions) and
                        (item.stride in self.stride_values) and
                        (item.padding in self.padding_values))
            elif isinstance(item, CNNrepository.Flatten):
                return ((item.in_channels in self.channel_dimensions) and
                        (item.input_size in self.higher_dimensions) and
                        (item.out_features in self.linear_feature_dimension_set))
            if isinstance(item, CNNrepository.Unflatten):
                return ((item.in_features in self.linear_feature_dimension_set) and
                        (item.out_channels in self.channel_dimensions) and
                        (item.output_size in self.higher_dimensions))
            else:
                return False

    @dataclass(frozen=True)
    class MSEloss:
        reduction: str = "mean"

    @dataclass(frozen=True)
    class L1Loss:
        reduction: str = "mean"

    # We will only have one loss functions for training a network, therefore we describe loss-functions not as labels,
    # but as a dependent product of loss-function-names and their valid parameter tuples.
    class LossFunction(Group):
        name = "Loss_Function"

        def iter_mseloss(self):
            yield CNNrepository.MSEloss(reduction="mean")
            yield CNNrepository.MSEloss(reduction="sum")
            #yield ODErepository.MSEloss(reduction="none")

        def iter_l1loss(self):
            yield CNNrepository.L1Loss(reduction="mean")
            yield CNNrepository.L1Loss(reduction="sum")
            #yield ODErepository.L1Loss(reduction="none")

        def __iter__(self):
            yield from self.iter_mseloss()
            yield from self.iter_l1loss()

        def __contains__(self, value):
            return (value is None or ((isinstance(value, CNNrepository.MSEloss) or isinstance(value, CNNrepository.L1Loss))
                    and (value.reduction in VALID_REDUCTIONS)))

    @dataclass(frozen=True)
    class Adam:
        learning_rate: float = 0.001
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-08
        weight_decay: float = 0.0
        amsgrad: bool = False
        maximize: bool = False
        capturable: bool = False
        differentiable: bool = False
        decoupled_weight_decay: bool = False

    # We will only have one optimizer for training a network, therefore we describe optimizers not as labels or
    # loss-functions, but as a dependent product of an optimizer name and its valid parameters.
    class Optimizer(Group):
        name = "Optimizer"

        def __init__(self, learning_rate_values):
            self.learning_rate_values = tuple(learning_rate_values)
            self.learning_rate_value_set = set(self.learning_rate_values)

        def __iter__(self):
            for lr in self.learning_rate_values:
                yield CNNrepository.Adam(learning_rate=lr)

        def __contains__(self, value):
            return value is None or (isinstance(value, CNNrepository.Adam) and
                                     (value.learning_rate in self.learning_rate_value_set))


    # To ensure normal forms under the above mentioned rewriting rules, we will model string diagrams as a
    # sequential composition of parallel composed components.
    # Para is a Wrapper for those components into a string diagram/computational graph and extends the label with the
    # ingoing and outgoing edges of a node.
    class Para(Group):
        name = "Para"

        def __init__(self, labels, dimensions):
            self.labels = tuple(labels) + (None,)
            self.label_set = set(self.labels)
            self.dimensions = tuple(dimensions) + (None,)
            self.dimension_set = set(self.dimensions)

        def __iter__(self):
            for l in self.labels:
                if l is not None:
                    for i in self.dimensions:
                        if i is not None:
                            for o in self.dimensions:
                                if o is not None:
                                    yield l, i, o
                                    if i == o:
                                        for n in range (0, i):
                                            m = i - n
                                            assert m > 0
                                            yield ("swap", n, m), i, o
            yield None

        def __contains__(self, value):
            try:
                return (value is None
                    #or value in ["linear", "sigmoid", "relu", "sharpness_sigmoid", "lte", "sum", "product"]
                    or (isinstance(value, tuple)
                                     and len(value) == 3
                                     and (value[0] in self.label_set
                                          or (value[0][0] == "swap"
                                              and (value[0][1] in self.dimension_set
                                                   or value[0][1] == 0)
                                              and value[0][2] in self.dimension_set))
                                     and value[1] in self.dimension_set
                                     and value[2] in self.dimension_set))
            except TypeError: # in case value is not hashable
                raise ValueError("The requested target type has arguments for the dataclasses, that are not part of the repo parameters!")

    # ParaTuples is then the parallel composition of nodes, constructing a graph with ingoing and outgoing edges as the
    # sum of the ingoing and outgoing edges of the parallel composed nodes.
    class ParaTuples(Group):
        name = "ParaTuples"

        def __init__(self, para, max_length=3):
            self.para = tuple(para)
            self.max_length = max_length
            self._iter_cache = None

        def __iter__(self):
            if self._iter_cache is None:
                result = {()}
                previous_layer = {()}
                for _ in range(1, self.max_length + 1):
                    current_layer = set()
                    for para in self.para:
                        for suffix in previous_layer:
                            current_layer.add((para,) + suffix)
                    result.update(current_layer)
                    previous_layer = current_layer
                self._iter_cache = tuple(result)
            yield from self._iter_cache

        def __contains__(self, value):
            return value is None or (isinstance(value, tuple) and all(True if v is None else v in self.para for v in value))

        # As ParaTuples defines all possible parallel compositions of components as literals, we can ensure normalforms
        # on the literal-level by normalizing them.
        def normalform(self, value) -> bool:
            """
            beside(swap(n, 0, n), swap(m, 0, m))
            ->
            swap(n+m, 0, n+m)
            """
            if value is None:
                return True # because synthesis enforces, that every variance for None will be in normal form
            for l, r in zip(value[:-1], value[1:]):
                if l is not None and r is not None and _matches_swap_zero_pair(l, r):
                    """
                    beside(swap(n, 0, n), swap(m, 0, m))
                    ->
                    swap(n+m, 0, n+m)
                    """
                    return False
            return True

        def normalize(self, value):
            while(not self.normalform(value)):
                new_value = value
                for index, (l, r) in enumerate(zip(value[:-1], value[1:])):
                    if l is None or r is None or not _matches_swap_zero_pair(l, r):
                        continue

                    left_parts = _swap_parts(l[0])
                    right_parts = _swap_parts(r[0])
                    if left_parts is None or right_parts is None:
                        continue

                    n = left_parts[1]
                    m = right_parts[1]
                    """
                    beside(swap(n, 0, n), swap(m, 0, m))
                    ->
                    swap(n+m, 0, n+m)
                    """
                    before_i = new_value[:index]
                    after_i = new_value[index + 2:]
                    new_value = before_i + ((("swap", 0, n + m), n + m, n + m),) + after_i
                    break
                value = new_value
            return value

    # Finally, ParaTupleTuples is the sequential composition of parallel composed nodes, ensuring that the numbers of
    # outgoing edges of the first graph and the ingoing edges of the second graph match.
    class ParaTupleTuples(Group):
        name = "ParaTupleTuples"

        def __init__(self, para_tuples):
            self.para_tuples = para_tuples

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value):
            return value is None or (isinstance(value, tuple) and all(True if v is None else v in self.para_tuples for v in value))

        # As for ParaTuples we ensure normalforms on the literal-level for ParaTupleTulpes as well by normalizing them.
        def normalform(self, value) -> bool:
            """
            The associativity laws are handled by the way we use python tuples.
            The abiding law is an invariance, because otherwise we couldn't use tuples of tuples.

            Therefore, we only need to check:
            - Neutrality of edges
            - Swap laws
            - Unique representation of parallel edges (swaps with n=0) (handled in ParaTuples)
            """
            if value is None:
                return True # because synthesis enforces, that every variance for None will be in normal form
            for l, r in zip(value[:-1], value[1:]):
                if l is not None and r is not None:
                    if not (self.para_tuples.normalform(l) and self.para_tuples.normalform(r)):
                        return False
                    if _is_singleton_tuple(l) and l[0] is not None:
                        label, i, o = l[0]
                        parts = _swap_parts(label)
                        if parts is not None:
                            m, n = parts
                            if m == 0:
                                """
                                before(edge(), x)
                                ->
                                x
                                """
                                return False
                            if _is_singleton_tuple(r) and r[0] is not None:
                                """
                                before(swap(m+n, m, n), swap(n+m, n, m))
                                ->
                                copy(m+n, edge())
                                """
                                right_label, right_i, right_o = r[0]
                                right_parts = _swap_parts(right_label)
                                if right_parts is not None:
                                    right_m, right_n = right_parts
                                    if right_m is not None and right_n is not None:
                                        if m == right_n and n == right_m:
                                            return False
                    if _is_singleton_tuple(r) and r[0] is not None:
                        """
                        before(x, edge())
                        ->
                        x
                        """
                        label, i, o = r[0]
                        if _is_swap_zero(label):
                                return False
                    if _is_pair_tuple(l) and _is_pair_tuple(r):
                        left_first, left_second = l
                        right_first, right_second = r
                        if left_first is not None and left_second is not None and right_first is not None and right_second is not None:
                            label_l_1, i_l_1, o_l_1 = left_first
                            label_l_2, i_l_2, o_l_2 = left_second
                            label_r_1, i_r_1, o_r_1 = right_first
                            label_r_2, i_r_2, o_r_2 = right_second
                            left_1 = _swap_parts(label_l_1)
                            left_2 = _swap_parts(label_l_2)
                            right_1 = _swap_parts(label_r_1)
                            right_2 = _swap_parts(label_r_2)
                            if left_1 is not None:
                                m, n = left_1
                                if left_2 is not None and left_2[0] == 0:
                                    p = left_2[1]
                                    if right_1 is not None and right_1[0] == 0:
                                        right_n = right_1[1]
                                        if right_2 is not None:
                                            right_m, right_p = right_2
                                            if m is not None and n is not None and p is not None and right_m is not None and right_n is not None and right_p is not None:
                                                if m == right_m and n == right_n and p == right_p:
                                                    """
                                                    before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
                                                    ->
                                                    swap(m + n + p, m, n+p)
                                                    """
                                                    return False
                            if left_1 is not None and left_1[0] == 0:
                                m = left_1[1]
                                if left_2 is not None:
                                    n, p = left_2
                                    if right_1 is not None:
                                        right_m, right_p = right_1
                                        if right_2 is not None and right_2[0] == 0:
                                            right_n = right_2[1]
                                            if m is not None and n is not None and p is not None and right_m is not None and right_n is not None and right_p is not None:
                                                if m == right_m and n == right_n and p == right_p:
                                                    """
                                                    before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
                                                    ->
                                                    swap(m + n + p, m+n, p)
                                                    """
                                                    return False

            for l, m, r in zip(value[:-2], value[1:-1], value[2:]):
                if l is not None and m is not None and r is not None:
                    if _is_singleton_tuple(l) and l[0] is not None:
                        left_label, left_i, left_o = l[0]
                        left_parts = _swap_parts(left_label)
                        if left_parts is not None:
                            left_m, left_n = left_parts
                            if _is_pair_tuple(m) and _is_singleton_tuple(r) and r[0] is not None:
                                mid_first, mid_second = m
                                right_label, right_i, right_o = r[0]
                                if mid_first is not None and mid_second is not None:
                                    mid_first_label, mid_n, mid_p = mid_first
                                    mid_second_label, mid_m, mid_q = mid_second
                                    right_parts = _swap_parts(right_label)
                                    if right_parts is not None:
                                        right_p, right_q = right_parts
                                        if left_m is not None and left_n is not None and mid_m is not None and mid_n is not None and mid_p is not None and mid_q is not None and right_p is not None and right_q is not None:
                                            if left_m == mid_m and left_n == mid_n and right_p == mid_p and right_q == mid_q:
                                                """
                                                before(swap(m + n, m, n), before(beside(x(n, p), y(m, q)), swap(p + q, p, q)))
                                                ->
                                                beside(y(m, q), x(n, p))
                                                """
                                                return False
            return True

        def normalize(self, value):
            if (not self.normalform(value)):
                value = tuple(map(self.para_tuples.normalize, value))
            while(not self.normalform(value)):
                new_value = value
                index = 0 # index in new_value
                for l, r in zip(value[:-1], value[1:]):
                    if _is_singleton_tuple(l):
                        label, i, o = l[0]
                        parts = _swap_parts(label)
                        if parts is not None:
                            m, n = parts
                            if m == 0:
                                """
                                before(edge(), x)
                                ->
                                x
                                """
                                # i < len(new_value) is an invariant, because len(zip(value[:-1], value[1:])) == len(value) - 1
                                before_i = new_value[:index]
                                after_i = new_value[index + 2:]
                                new_value = before_i + (r,) + after_i
                                break
                            if _is_singleton_tuple(r):
                                """
                                before(swap(m+n, m, n), swap(n+m, n, m))
                                ->
                                copy(m+n, edge())
                                """
                                right_label, right_i, right_o = r[0]
                                right_parts = _swap_parts(right_label)
                                if right_parts is not None:
                                    right_m, right_n = right_parts
                                    if m == right_n and n == right_m:
                                        before_i = new_value[:index]
                                        after_i = new_value[index + 2:]
                                        new_value = before_i + (((("swap", 0, n + m), n + m, n + m),),) + after_i
                                        break
                    if _is_singleton_tuple(r) and r[0] is not None:
                        """
                        before(x, edge())
                        ->
                        x
                        """
                        label, i, o = r[0]
                        if _is_swap_zero(label):
                            before_i = new_value[:index]
                            after_i = new_value[index + 2:]
                            new_value = before_i + (l,) + after_i
                            break
                    if _is_pair_tuple(l) and _is_pair_tuple(r):
                        left_first, left_second = l
                        right_first, right_second = r
                        if left_first and left_second and right_first and right_second:
                            label_l_1, i_l_1, o_l_1 = left_first
                            label_l_2, i_l_2, o_l_2 = left_second
                            label_r_1, i_r_1, o_r_1 = right_first
                            label_r_2, i_r_2, o_r_2 = right_second
                            left_1 = _swap_parts(label_l_1)
                            left_2 = _swap_parts(label_l_2)
                            right_1 = _swap_parts(label_r_1)
                            right_2 = _swap_parts(label_r_2)
                            if left_1 is not None:
                                m, n = left_1
                                if left_2 is not None and left_2[0] == 0:
                                    p = left_2[1]
                                    if right_1 is not None and right_1[0] == 0:
                                        right_n = right_1[1]
                                        if right_2 is not None:
                                            right_m, right_p = right_2
                                            if m == right_m and n == right_n and p == right_p:
                                                """
                                                before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
                                                ->
                                                swap(m + n + p, m, n+p)
                                                """
                                                before_i = new_value[:index]
                                                after_i = new_value[index + 2:]
                                                new_value = before_i + (((("swap", m, n + p), m + n + p, m + n + p),),) + after_i
                                                break
                            if left_1 is not None and left_1[0] == 0:
                                m = left_1[1]
                                if left_2 is not None:
                                    n, p = left_2
                                    if right_1 is not None:
                                        right_m, right_p = right_1
                                        if right_2 is not None and right_2[0] == 0:
                                            right_n = right_2[1]
                                            if m == right_m and n == right_n and p == right_p:
                                                """
                                                before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
                                                ->
                                                swap(m + n + p, m+n, p)
                                                """
                                                before_i = new_value[:index]
                                                after_i = new_value[index + 2:]
                                                new_value = before_i + (((("swap", m + n, p), m + n + p, m + n + p),),) + after_i
                                                break
                for l, m, r in zip(value[:-2], value[1:-1], value[2:]):
                    if _is_singleton_tuple(l):
                        left_label, left_i, left_o = l[0]
                        left_parts = _swap_parts(left_label)
                        if left_parts is not None:
                            left_m, left_n = left_parts
                            if _is_pair_tuple(m) and _is_singleton_tuple(r):
                                mid_first, mid_second = m
                                right_label, right_i, right_o = r[0]
                                mid_first_label, mid_n, mid_p = mid_first
                                mid_second_label, mid_m, mid_q = mid_second
                                right_parts = _swap_parts(right_label)
                                if right_parts is not None:
                                    right_p, right_q = right_parts
                                    if left_m == mid_m and left_n == mid_n and right_p == mid_p and right_q == mid_q:
                                        """
                                        before(swap(m + n, m, n), before(beside(x(n, p), y(m, q)), swap(p + q, p, q)))
                                        ->
                                        beside(y(m, q), x(n, p))
                                        """
                                        before_i = new_value[:index]
                                        after_i = new_value[index + 3:]
                                        new_value = before_i + ((mid_second, mid_first),) + after_i
                                        break
                value = new_value
            return value


    # To ensure normalforms on the term-level we have to implement a term-predicate for each term rewriting rule that
    # we can't enforce by clever typing of the combinators. The term-predicates are defined on Tree and have to ensure,
    # that the left-hand side of the rewrite rule is forbidden. This ensures, that no term will be synthesized, that
    # matches a left-hand side of a rewrite rule.

    @staticmethod
    def swaplaw1(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
        ->
        beside(y(m,q),x(n,p))

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """

        if _tree_root_contains(head, "beside_singleton") and _tree_root_contains(tail, "before_cons"):
            _require_children_count(head, 5)
            _require_children_count(tail, 9)

            left_term = _tree_child(head, 4)
            right_head = _tree_child(tail, 7)
            right_tail = _tree_child(tail, 8)

            if _tree_root_is(left_term, "swap") and _tree_root_contains(right_head, "beside_cons") and _tree_root_contains(right_tail, "before_singleton"):
                _require_children_count(left_term, 19)
                _require_children_count(right_head, 11)
                _require_children_count(right_tail, 6)

                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]

                right_head_tail = right_head.children[10]
                right_tail_term = right_tail.children[5]

                if _tree_root_contains(right_head_tail, "beside_singleton") and _tree_root_contains(right_tail_term, "beside_singleton"):
                    _require_children_count(right_head_tail, 5)
                    _require_children_count(right_tail_term, 5)

                    y_m = right_head_tail.children[0]
                    y_q = right_head_tail.children[1]
                    right_swap = right_tail_term.children[4]

                    if _tree_root_is(right_swap, "swap"):
                        _require_children_count(right_swap, 19)
                        p = right_swap.children[1]
                        q = right_swap.children[2]
                        if m == y_m and n == x_n and p == x_p and q == y_q:
                            return False

            elif _tree_root_is(head, "swap") and _tree_root_contains(right_head, "beside_cons") and _tree_root_contains(right_tail, "before_cons"):
                left_term = _tree_child(head, 4)
                _require_children_count(left_term, 19)
                _require_children_count(right_head, 11)
                _require_children_count(right_tail, 9)

                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[10]
                right_tail_head = right_tail.children[7]

                if _tree_root_contains(right_head_tail, "beside_singleton") and _tree_root_contains(right_tail_head, "beside_singleton"):
                    _require_children_count(right_head_tail, 5)
                    _require_children_count(right_tail_head, 5)

                    y_m = right_head_tail.children[0]
                    y_q = right_head_tail.children[1]
                    right_swap = right_tail_head.children[4]

                    if _tree_root_is(right_swap, "swap"):
                        _require_children_count(right_swap, 19)
                        p = right_swap.children[1]
                        q = right_swap.children[2]
                        if m == y_m and n == x_n and p == x_p and q == y_q:
                            return False
        return True

    @staticmethod
    def swaplaw2(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
        ->
        swap(m + n + p, m, n+p)

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched

        """

        if _tree_root_contains(head, "beside_cons") and _tree_root_contains(tail, "before_singleton"):
            _require_children_count(head, 11)
            _require_children_count(tail, 6)

            left_head = _tree_child(head, 9)
            left_tail = _tree_child(head, 10)
            right_term = _tree_child(tail, 5)

            if _tree_root_is(left_head, "swap") and _tree_root_contains(left_tail, "beside_singleton") and _tree_root_contains(right_term, "beside_cons"):
                _require_children_count(left_head, 19)
                _require_children_count(left_tail, 5)
                _require_children_count(right_term, 11)

                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4]  # swap(p, 0, p)
                right_head = right_term.children[9]  # swap(n, 0, n)
                right_tail = right_term.children[10]

                if _tree_root_is(left_tail_term, "edges") and _tree_root_is(right_head, "edges") and _tree_root_contains(right_tail, "beside_singleton"):
                    _require_children_count(left_tail_term, 17)
                    _require_children_count(right_head, 17)
                    _require_children_count(right_tail, 5)

                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]
                    right_tail_term = right_tail.children[4]  # swap(m+p, m, p)

                    if _tree_root_is(right_tail_term, "swap") and n == right_n:
                        _require_children_count(right_tail_term, 19)
                        right_m = right_tail_term.children[1]
                        right_p = right_tail_term.children[2]
                        if m == right_m and p == right_p:
                            return False

        elif _tree_root_contains(head, "beside_cons") and _tree_root_contains(tail, "before_cons"):
            _require_children_count(head, 11)
            _require_children_count(tail, 9)

            left_head = _tree_child(head, 9)
            left_tail = _tree_child(head, 10)
            right_term = _tree_child(tail, 7)

            if _tree_root_is(left_head, "swap") and _tree_root_contains(left_tail, "beside_singleton") and _tree_root_contains(right_term, "beside_cons"):
                _require_children_count(left_head, 19)
                _require_children_count(left_tail, 5)
                _require_children_count(right_term, 11)

                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4]  # swap(p, 0, p)
                right_head = right_term.children[9]  # swap(n, 0, n)
                right_tail = right_term.children[10]

                if _tree_root_is(left_tail_term, "edges") and _tree_root_is(right_head, "edges") and _tree_root_contains(right_tail, "beside_singleton"):
                    _require_children_count(left_tail_term, 17)
                    _require_children_count(right_head, 17)
                    _require_children_count(right_tail, 5)

                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]
                    right_tail_term = right_tail.children[4]  # swap(m+p, m, p)

                    if _tree_root_is(right_tail_term, "swap") and n == right_n:
                        _require_children_count(right_tail_term, 19)
                        right_m = right_tail_term.children[1]
                        right_p = right_tail_term.children[2]
                        if m == right_m and p == right_p:
                            return False
        return True

    @staticmethod
    def swaplaw3(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(swap(m+n, m, n), swap(n+m, n, m))
        ->
        copy(m+n, edge())

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """
        if _tree_root_contains(head, "beside_singleton") and _tree_root_contains(tail, "before_singleton"):
            _require_children_count(head, 5)
            _require_children_count(tail, 6)

            left_term = _tree_child(head, 4)
            right_term = _tree_child(tail, 5)

            if _tree_root_is(left_term, "swap") and _tree_root_contains(right_term, "beside_singleton"):
                _require_children_count(left_term, 19)
                _require_children_count(right_term, 5)

                m = left_term.children[1]
                n = left_term.children[2]
                right_beside = right_term.children[4]

                if _tree_root_is(right_beside, "swap"):
                    _require_children_count(right_beside, 19)
                    right_n = right_beside.children[1]
                    right_m = right_beside.children[2]
                    if m == right_m and n == right_n:
                        return False

        elif _tree_root_contains(head, "beside_singleton") and _tree_root_contains(tail, "before_cons"):
            _require_children_count(head, 5)
            _require_children_count(tail, 9)

            left_term = _tree_child(head, 4)
            right_term = _tree_child(tail, 7)

            if _tree_root_is(left_term, "swap") and _tree_root_contains(right_term, "beside_singleton"):
                _require_children_count(left_term, 19)
                _require_children_count(right_term, 5)

                m = left_term.children[1]
                n = left_term.children[2]
                right_beside = right_term.children[4]

                if _tree_root_is(right_beside, "swap"):
                    _require_children_count(right_beside, 19)
                    right_n = right_beside.children[1]
                    right_m = right_beside.children[2]
                    if m == right_m and n == right_n:
                        return False
        return True

    @staticmethod
    def swaplaw4(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
        ->
        swap(m + n + p, m+n, p)

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """
        if _tree_root_contains(head, "beside_cons") and _tree_root_contains(tail, "before_singleton"):
            _require_children_count(head, 11)
            _require_children_count(tail, 6)

            left_head = _tree_child(head, 9)
            left_tail = _tree_child(head, 10)
            right_term = _tree_child(tail, 5)

            if _tree_root_is(left_head, "edges") and _tree_root_contains(left_tail, "beside_singleton") and _tree_root_contains(right_term, "beside_cons"):
                _require_children_count(left_head, 17)
                _require_children_count(left_tail, 5)
                _require_children_count(right_term, 11)

                m = left_head.children[0]
                left_tail_term = left_tail.children[4]
                right_head = right_term.children[9]
                right_tail = right_term.children[10]

                if _tree_root_is(left_tail_term, "swap") and _tree_root_is(right_head, "swap") and _tree_root_contains(right_tail, "beside_singleton"):
                    _require_children_count(left_tail_term, 19)
                    _require_children_count(right_head, 19)
                    _require_children_count(right_tail, 5)

                    n = left_tail_term.children[1]
                    p = left_tail_term.children[2]
                    right_m = right_head.children[1]
                    right_p = right_head.children[2]
                    right_tail_term = right_tail.children[4]

                    if _tree_root_is(right_tail_term, "edges") and m == right_m and p == right_p:
                        _require_children_count(right_tail_term, 17)
                        right_n = right_tail_term.children[0]
                        if n == right_n:
                            return False

        elif _tree_root_contains(head, "beside_cons") and _tree_root_contains(tail, "before_cons"):
            _require_children_count(head, 11)
            _require_children_count(tail, 9)

            left_head = _tree_child(head, 9)
            left_tail = _tree_child(head, 10)
            right_term = _tree_child(tail, 7)

            if _tree_root_is(left_head, "edges") and _tree_root_contains(left_tail, "beside_singleton") and _tree_root_contains(right_term, "beside_cons"):
                _require_children_count(left_head, 17)
                _require_children_count(left_tail, 5)
                _require_children_count(right_term, 11)

                m = left_head.children[0]
                left_tail_term = left_tail.children[4]
                right_head = right_term.children[9]
                right_tail = right_term.children[10]

                if _tree_root_is(left_tail_term, "swap") and _tree_root_is(right_head, "swap") and _tree_root_contains(right_tail, "beside_singleton"):
                    _require_children_count(left_tail_term, 19)
                    _require_children_count(right_head, 19)
                    _require_children_count(right_tail, 5)

                    n = left_tail_term.children[1]
                    p = left_tail_term.children[2]
                    right_m = right_head.children[1]
                    right_p = right_head.children[2]
                    right_tail_term = right_tail.children[4]

                    if _tree_root_is(right_tail_term, "edges") and m == right_m and p == right_p:
                        _require_children_count(right_tail_term, 17)
                        right_n = right_tail_term.children[0]
                        if n == right_n:
                            return False
        return True

    # for all rewrite rules that aren't covered by term-predicates the left-hand sides will be forbidden by clever
    # type assignments for the combinators in the following specification, that forbid the construction of terms that
    # match the left-hand sides of the rewrite rules.
    def specification(self):
        labels = self.Label(self.dimensions, self.linear_feature_dimensions, self.constant_values,
                            self.channel_dimensions, self.kernel_dimensions, self.stride_values,
                            self.padding_values, self.higher_dimensions)
        para_labels = self.Para(labels, self.dimensions)
        #print("linear" in para_labels)
        paratuples = self.ParaTuples(para_labels, max_length=max(self.dimensions))
        paratupletuples = self.ParaTupleTuples(paratuples)
        dimension = DataGroup("dimension", self.dimensions)
        dimension_with_None = DataGroup("dimension_with_None", list(self.dimensions) + [None])
        feature_dimension = DataGroup("feature_dimension", self.linear_feature_dimensions)
        higher_dimension = DataGroup("higher_dimension",
                                     [(c, h, w) for c in self.channel_dimensions for h, w in self.higher_dimensions])
        higher_dimension_with_None = DataGroup("higher_dimension_with_None",
                                               [(c, h, w) for c in self.channel_dimensions
                                                for h, w in self.higher_dimensions] + [None])
        channel_dimension = DataGroup("channel_dimension", self.channel_dimensions)
        loss_function = self.LossFunction()
        optimizer = self.Optimizer(self.learning_rate_values)
        epochs = DataGroup("epochs", self.n_epoch_values)

        return {
            # atomic components are nodes and edges
            # but edges are a special case of swaps (swaps, that do not change anything)
            # so we can just define edges as a special case of swaps here
            #"edge": Constructor("DAG", Constructor("input", Literal(1))
            #                    & Constructor("input", Literal(None))
            #                    & Constructor("output", Literal(1))
            #                    & Constructor("output", Literal(None))
            #                    & Constructor("structure", Literal(((("swap", Literal(0), Literal(1)),),)))
            #                    & Constructor("structure", Literal(((None,),)))),
            # (m parallel) edges are swaps with n=0
            "edges": SpecificationBuilder()
            .parameter("io", dimension)
            .parameter("para1", para_labels, lambda v: [(("swap", 0, v["io"]), v["io"], v["io"])])
            .parameter("para2", para_labels, lambda v: [(("swap", 0, None), v["io"], v["io"])])
            .parameter("para3", para_labels, lambda v: [(("swap", None, v["io"]), v["io"], v["io"])])
            .parameter("para4", para_labels, lambda v: [(("swap", None, None), v["io"], v["io"])])
            .parameter("para5", para_labels, lambda v: [(("swap", 0, v["io"]), v["io"], None)])
            .parameter("para6", para_labels, lambda v: [(("swap", 0, None), v["io"], None)])
            .parameter("para7", para_labels, lambda v: [(("swap", None, v["io"]), v["io"], None)])
            .parameter("para8", para_labels, lambda v: [(("swap", None, None), v["io"], None)])
            .parameter("para9", para_labels, lambda v: [(("swap", 0, v["io"]), None, v["io"])])
            .parameter("para10", para_labels, lambda v: [(("swap", 0, None), None, v["io"])])
            .parameter("para11", para_labels, lambda v: [(("swap", None, v["io"]), None, v["io"])])
            .parameter("para12", para_labels, lambda v: [(("swap", None, None), None, v["io"])])
            .parameter("para13", para_labels, lambda v: [(("swap", 0, v["io"]), None, None)])
            .parameter("para14", para_labels, lambda v: [(("swap", 0, None), None, None)])
            .parameter("para15", para_labels, lambda v: [(("swap", None, v["io"]), None, None)])
            .parameter("para16", para_labels, lambda v: [(("swap", None, None), None, None)])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                & Constructor("structure", Var("para8"))
                                & Constructor("structure", Var("para9"))
                                & Constructor("structure", Var("para10"))
                                & Constructor("structure", Var("para11"))
                                & Constructor("structure", Var("para12"))
                                & Constructor("structure", Var("para13"))
                                & Constructor("structure", Var("para14"))
                                & Constructor("structure", Var("para15"))
                                & Constructor("structure", Var("para16"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("ID")
                    ),

            "swap": SpecificationBuilder()
            .parameter("io", dimension)
            .parameter("n", dimension, lambda v: range(1, v["io"]))
            .parameter("m", dimension, lambda v: [v["io"] - v["n"]]) # m > 0
            .parameter("para1", para_labels, lambda v: [(("swap", v["n"], v["m"]), v["io"], v["io"])])
            .parameter("para2", para_labels, lambda v: [(("swap", v["n"], None), v["io"], v["io"])])
            .parameter("para3", para_labels, lambda v: [(("swap", None, v["m"]), v["io"], v["io"])])
            .parameter("para4", para_labels, lambda v: [(("swap", None, None), v["io"], v["io"])])
            .parameter("para5", para_labels, lambda v: [(("swap", v["n"], v["m"]), v["io"], None)])
            .parameter("para6", para_labels, lambda v: [(("swap", v["n"], None), v["io"], None)])
            .parameter("para7", para_labels, lambda v: [(("swap", None, v["m"]), v["io"], None)])
            .parameter("para8", para_labels, lambda v: [(("swap", None, None), v["io"], None)])
            .parameter("para9", para_labels, lambda v: [(("swap", v["n"], v["m"]), None, v["io"])])
            .parameter("para10", para_labels, lambda v: [(("swap", v["n"], None), None, v["io"])])
            .parameter("para11", para_labels, lambda v: [(("swap", None, v["m"]), None, v["io"])])
            .parameter("para12", para_labels, lambda v: [(("swap", None, None), None, v["io"])])
            .parameter("para13", para_labels, lambda v: [(("swap", v["n"], v["m"]), None, None)])
            .parameter("para14", para_labels, lambda v: [(("swap", v["n"], None), None, None)])
            .parameter("para15", para_labels, lambda v: [(("swap", None, v["m"]), None, None)])
            .parameter("para16", para_labels, lambda v: [(("swap", None, None), None, None)])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                & Constructor("structure", Var("para8"))
                                & Constructor("structure", Var("para9"))
                                & Constructor("structure", Var("para10"))
                                & Constructor("structure", Var("para11"))
                                & Constructor("structure", Var("para12"))
                                & Constructor("structure", Var("para13"))
                                & Constructor("structure", Var("para14"))
                                & Constructor("structure", Var("para15"))
                                & Constructor("structure", Var("para16"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "linear_layer": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_linear()))
            .parameter("i", dimension, lambda v: [v["l"].in_features])
            .parameter("o", dimension, lambda v: [v["l"].out_features])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                #& Constructor("structure", Literal("linear"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "conv2d": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_conv2d()))
            .parameter("i", higher_dimension, lambda v: [(v["l"].in_channels, v["l"].input_size[0], v["l"].input_size[1])])
            .parameter("o", higher_dimension, lambda v: [(v["l"].out_channels, v["l"].output_size[0], v["l"].output_size[1])])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                # & Constructor("structure", Literal("linear"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "maxpool2d": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_maxpool2d()))
            .parameter("i", higher_dimension,
                       lambda v: [(v["l"].in_channels, v["l"].input_size[0], v["l"].input_size[1])])
            .parameter("o", higher_dimension,
                       lambda v: [(v["l"].out_channels, v["l"].output_size[0], v["l"].output_size[1])])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                # & Constructor("structure", Literal("linear"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "flatten": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_flatten()))
            .parameter("i", higher_dimension,
                       lambda v: [(v["l"].in_channels, v["l"].input_size[0], v["l"].input_size[1])])
            .parameter("o", dimension, lambda v: [v["l"].out_features])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                # & Constructor("structure", Literal("linear"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "unflatten": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_unflatten()))
            .parameter("i", dimension, lambda v: [v["l"].in_features])
            .parameter("o", higher_dimension,
                       lambda v: [(v["l"].out_channels, v["l"].output_size[0], v["l"].output_size[1])])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                # & Constructor("structure", Literal("linear"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "sigmoid": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_sigmoid()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [v["i"]])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                #& Constructor("structure", Literal("sigmoid"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "relu": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_relu()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [v["i"]])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                #& Constructor("structure", Literal("relu"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "tanh": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_tanh()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [v["i"]])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                # & Constructor("structure", Literal("relu"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "sum": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_sum()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [1])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                #& Constructor("structure", Literal("sum"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "product": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_product()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [1])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                #& Constructor("structure", Literal("product"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "copy": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_copy()))
            .parameter("i", dimension, lambda v: [1])
            .parameter("o", dimension, lambda v: [v["l"].out_dimension])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                # & Constructor("structure", Literal("product"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            # TODO: add CNN-dimension to composition

            "beside_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratuples)
            .parameter_constraint(lambda v: v["ls"] is None or len(v["ls"]) == 1)
            .parameter("para", para_labels, lambda v: [None] if v["ls"] is None else [v["ls"][0]])
            .parameter_constraint(lambda v: (len(v["para"]) == 3 and
                                             (v["para"][1] == v["i"] if v["para"][1] is not None else True) and
                                             (v["para"][2] == v["o"] if v["para"][2] is not None else True)
                                             ) if v["para"] is not None else True)
            .suffix(
                ((Constructor("DAG_component",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("para")))
                     & Constructor("non_ID")
                  )
                 **
                 (Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                #& Constructor("structure", Literal(None))
                                )
                  & Constructor("non_ID") & Constructor("last", Constructor("non_ID"))
                  )
                 )
                &
                ((Constructor("DAG_component",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("para")))
                     & Constructor("ID")
                     )
                 **
                 (Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                #& Constructor("structure", Literal(None))
                                )
                     & Constructor("ID")
                     )
                 )),

            "beside_cons": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("i1", dimension)
            .parameter("i2", dimension, lambda v: [v["i"] - v["i1"]])
            .parameter("o", dimension)
            .parameter("o1", dimension)
            .parameter("o2", dimension, lambda v: [v["o"] - v["o1"]])
            .parameter("ls", paratuples)
            .parameter_constraint(lambda v: v["ls"] is None or len(v["ls"]) > 1)
            .parameter("head", para_labels, lambda v: [None] if v["ls"] is None else [v["ls"][0]])
            .parameter_constraint(lambda v: v["head"] is None or (len(v["head"]) == 3 and
                                                                  (v["head"][1] == v["i1"] or v["head"][1] is None) and
                                                                  (v["head"][2] == v["o1"] or v["head"][2] is None)))
            .parameter("tail", paratuples, lambda v: [None] if v["ls"] is None else [v["ls"][1:]])
            .suffix(
                    ((Constructor("DAG_component",
                                       Constructor("input", Var("i1"))
                                       & Constructor("output", Var("o1"))
                                       & Constructor("structure", Var("head")))
                      & Constructor("ID"))
                    **
                    (Constructor("DAG_parallel",
                                       Constructor("input", Var("i2"))
                                       & Constructor("output", Var("o2"))
                                       & Constructor("structure", Var("tail")))
                     & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     **
                     (Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                #& Constructor("structure", Literal(None))
                                )
                      & Constructor("non_ID") & Constructor("last", Constructor("ID")))
                     )
                    &
                    ((Constructor("DAG_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("structure", Var("head")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i2"))
                                  & Constructor("output", Var("o2"))
                                  & Constructor("structure", Var("tail")))
                      & Constructor("ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("structure", Var("ls"))
                                  #& Constructor("structure", Literal(None))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    &
                    ((Constructor("DAG_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("structure", Var("head")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i2"))
                                  & Constructor("output", Var("o2"))
                                  & Constructor("structure", Var("tail")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("structure", Var("ls"))
                                  #& Constructor("structure", Literal(None))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    ),

            # normalization is already done at learner combinator and may be removed here, but this would require refactoring of term predicates...
            "before_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("request", paratupletuples)
            .parameter("ls", paratupletuples, lambda v: [paratupletuples.normalize(v["request"])])
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) == 1)
            .parameter("ls1", paratuples, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: v["ls1"] is None or
                                            (
                                                    (
                                                        v["i"] == sum([t[1] for t in v["ls1"]])
                                                        if None not in [t for t in v["ls1"]]
                                                           and None not in [t[1] for t in v["ls1"]]
                                                        else v["i"] > sum([t[1] for t in v["ls1"]
                                                                           if t is not None
                                                                           and t[1] is not None])
                                                    )
                                                    and
                                                    (
                                                        v["o"] == sum([t[2] for t in v["ls1"]])
                                                        if None not in [t for t in v["ls1"]]
                                                           and None not in [t[2] for t in v["ls1"]]
                                                        else v["o"] > sum([t[2] for t in v["ls1"]
                                                                           if t is not None
                                                                           and t[2] is not None]))
                                            )
                                  )
            .argument("x", Constructor("DAG_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("ls1"))) & Constructor("non_ID"))
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("request"))
                                )),

            # normalization is already done at learner combinator and may be removed here, but this would require refactoring of term predicates...
            "before_cons": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("j", dimension)
            .parameter("o", dimension)
            .parameter("request", paratupletuples)
            .parameter("ls", paratupletuples, lambda v: [paratupletuples.normalize(v["request"])])
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) > 1)
            .parameter("head", paratuples, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: v["head"] is None or
                                            (
                                                (v["i"] == sum([t[1] for t in v["head"]])
                                                 if None not in [t for t in v["head"]]
                                                    and None not in [t[1] for t in v["head"]]
                                                 else v["i"] > sum([t[1] for t in v["head"]
                                                                    if t is not None and t[1] is not None]))
                                                and (v["j"] == sum([t[2] for t in v["head"]])
                                                     if None not in [t for t in v["head"]]
                                                        and None not in [t[2] for t in v["head"]]
                                                     else v["j"] > sum([t[2] for t in v["head"]
                                                                        if t is not None and t[2] is not None]))
                                            )
                                  )
            .parameter("tail", paratupletuples, lambda v: [v["ls"][1:]])
            .parameter_constraint(lambda v: v["tail"] is None or
                                            (
                                                    (len(v["tail"]) > 0) and
                                                    (
                                                            v["tail"][0] is None or
                                                            (
                                                                v["j"] == sum([t[1] for t in v["tail"][0]])
                                                                if None not in [t for t in v["tail"][0]]
                                                                   and None not in [t[1] for t in v["tail"][0]]
                                                                else v["j"] > sum([t[1] for t in v["tail"][0]
                                                                                   if t is not None
                                                                                   and t[1] is not None])
                                                             )
                                                    ) and
                                                    (
                                                            v["tail"][-1] is None or
                                                            (v["o"] == sum([t[2] for t in v["tail"][-1]])
                                                             if None not in [t for t in v["tail"][-1]]
                                                                and None not in [t[2] for t in v["tail"][-1]]
                                                             else v["o"] > sum([t[2] for t in v["tail"][-1]
                                                                                if t is not None
                                                                                and t[2] is not None]))
                                                    )
                                            )
                                  )
            .argument("x", Constructor("DAG_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("j"))
                                       & Constructor("structure", Var("head"))) & Constructor("non_ID"))
            .argument("y", Constructor("DAG",
                                       Constructor("input", Var("j"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("tail"))))
            .constraint(lambda v: self.swaplaw1(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw2(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw3(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw4(v["x"], v["y"]))
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("request")))),

            "mse_loss": SpecificationBuilder()
            .parameter("loss", loss_function, lambda v: list(loss_function.iter_mseloss()))
            .suffix(Constructor("Loss", Constructor("type", Var("loss")) & Constructor("type", Literal(None)))),

            "l1loss": SpecificationBuilder()
            .parameter("loss", loss_function, lambda v: list(loss_function.iter_l1loss()))
            .suffix(Constructor("Loss", Constructor("type", Var("loss")) & Constructor("type", Literal(None)))),

            "adam_optimizer": SpecificationBuilder()
            .parameter("optimizer", optimizer)
            .suffix(Constructor("Optimizer", Constructor("type", Var("optimizer")) & Constructor("type", Literal(None)))),

            "learner": SpecificationBuilder()
            .parameter("i", dimension_with_None)
            .parameter("o", dimension_with_None)
            .parameter("request", paratupletuples)
            .parameter("ls", paratupletuples, lambda v: [paratupletuples.normalize(v["request"])])
            .parameter("epochs", epochs)
            .parameter("loss", loss_function)
            .parameter("opti", optimizer)
            .argument("loss_f", Constructor("Loss", Constructor("type", Var("loss"))))
            .argument("optimizer", Constructor("Optimizer", Constructor("type", Var("opti"))))
            .argument("model", Constructor("DAG",
                                           Constructor("input", Var("i"))
                                           & Constructor("output", Var("o"))
                                           & Constructor("structure", Var("ls"))))
            .suffix(Constructor("Learner", Constructor("DAG",
                                                       Constructor("input", Var("i"))
                                                       & Constructor("output", Var("o"))
                                                       & Constructor("structure", Var("request"))
                                                       )
                                & Constructor("Loss", Constructor("type", Var("loss")))
                                & Constructor("Optimizer", Constructor("type", Var("opti")))
                                & Constructor("epochs", Var("epochs"))
                                )
                    )
        }



