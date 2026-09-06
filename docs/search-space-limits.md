# What a search space has to admit

The loop draws terms, and everything below is about that. A search space that does not admit the
draw a component needs stops the run, and it usually stops it after the construction has been paid
for rather than before.

## A sampler without a bound is not a sampler

Emptiness of the languages a repository with term predicates describes is undecidable, and within
a bound the question changes. So a sampler
maps a query to a stream of terms **within a bound**, and every clause that reacts to a stream
delivering nothing reacts to a query that halted, never to an emptiness test. A caller who gives a
sampler no bound is wrong by that definition, whatever the sampler then does.

`BayesianOptimization` therefore takes a `Sampler` object rather than two numbers. Passing `None`
builds `SizeUniformSampler(100, Random(seed))`, which is a placeholder for toy spaces and the first
thing to replace on a real one.

## Three bounds that are not the same quantity

The bound of `SizeUniformSampler` is a term size, the number of symbols in the term. Size-uniform
sampling draws a realized size uniformly and then an inhabitant of that size uniformly, both within
the bound, so what the bound admits is what the sampler counts.

The bound of `DepthBoundedRandomSampler` is a depth, the longest path from the root to a leaf.

Neither is cosy's engine parameter `max_depth`, which bounds the open positions of a goal. A
subtree leaves that measurement as soon as it grounds, so `max_depth` is not a statement about the
terms produced at all.

Confusing the first two is not loud. A term whose combinators take two arguments can hold up to
`2^(d+1) - 1` symbols at depth `d`, so the two bounds do not describe comparable sets. The recorded
VGG-scale run in `recorded_runs/` drew at a depth bound of 1000, while the size-uniform default of
the same driver is a size bound of 200, `DEFAULT_SIZE_BOUND` in
`cnn_damg_nas/cnn_damg_experiment_utils.py`.

## What the size bound really constrains is the initial dataset

On a real space the size bound is a statement about which terms can appear in the initial dataset
at all. The sampler stratifies over the sizes the space realizes within that bound, so a bound
below the sizes a space realizes leaves the design drawing from a part of the space rather than
from the space. The CNN driver's default is 200, and on the length-2 CNN space with the USPS
parameter sets that
bound admits every term the space realizes: 55 occupied sizes from 62 to 199 and 41 385 472
inhabitants in all. There the bound is a formality. Where it stops being one is the paragraph below.

Below every size a space realizes, the design is not narrow but empty, and the failure arrives late.
The VGGM cell of the CIFAR-10 driver is that case at the shipped default. Measured on one laptop:
the size-uniform construction takes 57 seconds and the first draw counts for a further 58 before
the stream ends without a term. The 80 terms of the recorded run have sizes 486 to 795, so a bound
of 200 admits none of them, and the two minutes are spent before anything says so.

## Counting from the program applies only where no predicate reads a hole

`SizeUniformSampler` counts. It can count from a materialized search tree, `counting="tree"`, or
from the program, `counting="table"`, and the second is what makes a realistic space affordable.

The table applies only where no predicate of the repository reads a hole. For several holes the
residual of a partial term need not be the product of the single-hole residuals: a predicate that
relates two holes leaves a subterm admissible at the one hole only for certain fillings of the
other, so the residual there is a genuine relation and no table indexed by the non-terminal can be
right about it. `cosy.search.counting.decomposable_or_raise` names those clauses and refuses rather
than returning counts that are quietly too large.

The CNN repository is such a case. It states four swap laws as term predicates over two sibling
holes of `before_cons`, and at structure length 2 with the USPS parameter sets 8 of its clauses
carry one, every one of them a `before_cons` instance.

## Compiling the predicates away is a trade, not a gain

`recognizable_cnn_damg_repo.py` and `recognizable_damg_repo.py` state the same four laws as a
finite abstraction with a relation on its values, which is the form
`cosy.search.determinize.determinize` can push into the non-terminals. What comes out derives
exactly the terms the original derives, along exactly one branch each, and carries no predicate
over a hole, so the table applies to it.

What it costs is the size of the product. On the length-2 CNN space with the USPS parameter sets,
307 non-terminals and 856 rules become a product of 1044 states with 2605 rules. The program is now
countable and it is larger, and which of the two a run can afford is a property of the repository
and its parameter sets rather than of the method. `DETERMINIZATION_STATE_LIMIT` in
`cnn_damg_nas/cnn_damg_experiment_utils.py` is `1_000_000` and exists because the fixpoint has to
be stopped somewhere.

How far that product can grow is not read off the small case. On the VGGM cell the same
construction turns 641 non-terminals and 3912 rules into 27 337 states with 626 169 rules, which is
a factor of 160 on the rules where the small case saw a factor of 3. The state limit is there for
the direction that number is pointing in.

That the abstraction exists and that the product is affordable are two different statements. The
first is a property of the laws, the second a measurement per configuration.

Call each abstraction factory once per repository, at module level. The determinization collects
the distinct abstractions a program states and builds the product over them, and it tells two of
them apart by identity, which for a closure means the object. Four constraints naming four
separately built closures become four axes of a product where one would do. The four laws of these
repositories share one abstraction object, and `tests/test_recognizable_cnn_repo.py` asserts that
the compiled program carries exactly that one, because the state count above assumes it.

## One decision, both halves together

`build_search` in `cnn_damg_experiment_utils.py` returns the program a run searches together with
the sampler that fits it, because choosing them apart is how a run breaks. `sampling="size-uniform"`
determinizes and hands back a counting sampler over the product. `sampling="depth-bounded"` hands
back a `DepthBoundedRandomSampler`, which never counts, and it reports no determinization in its
provenance. Both modes hand back the coupled program as the program the loop searches. Only the
sampler ever sees the determinized one, because counting is the only thing it is better at.

Asking for size-uniform sampling on the repository whose laws are still term predicates raises
rather than counting something else.

Where a space can pay neither counting construction, the depth-bounded sampler is the only draw
left, and with it the size-uniform initial design is gone too. There is no counting-free stratified
design in this package, and a run in that situation draws its initial dataset from the depth-bounded
stream. That is what the recorded VGG-scale run did, and its configuration records both halves.

## A prefix of one stream, and one query object

A design of `mu_0` members is a prefix of one stream rather than `mu_0` draws. `distinct_prefix` in
`initial_sampling.py` takes it that way, because each call re-poses the query, and on a realistic
space that is what a draw costs. A stream that ends before the design is full raises a
`RuntimeError` naming how many distinct inhabitants it produced, rather than returning a short
design.

`SizeUniformSampler` softens the second half of that by caching its weighted construction, but it
keys the cache by query **identity** rather than by equality, since comparing a partial-term query
structurally would cost more than the lookup saves. So the caching only reaches a caller who passes
one and the same query object. `BayesianOptimization` mints one and shares it between the
initializer, the evolutionary search and the duplicate fallback, and exposes it as `query` so that a
caller drawing a paired baseline of their own draws through it too. A caller who builds a fresh
query per draw pays the whole counting construction again on every draw, where a caller who reuses
the query pays it once. On a space where that construction takes minutes, which is what the CNN
cells are, the difference is the difference between a run and no run.

## Where the term size carries no information

Size-uniform sampling stratifies along term size, which is the one canonical axis a search space
has. On a space whose terms differ mainly in constant arguments, that axis carries almost nothing.

Measured on the length-2 CNN space with the USPS parameter sets: within a size bound of 83 the space
realizes three sizes and no more, 62 with 400 inhabitants, 71 with 24 and 83 with 2312, which
`tests/test_recognizable_cnn_repo.py` pins as a row. Stratifying over three values is not a spread
over the space, and it is the reason a space of this shape is a candidate for the kernel-diverse
design or for a depth-bounded draw rather than for the default. Stratifying over three values is not
a spread over the space, and it is the reason a space of this shape is a candidate for the
kernel-diverse design or for a depth-bounded draw rather than for the default.
