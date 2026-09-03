"""Checks that decide a distributional claim rather than a single behavior.

A statistical test is only worth as much as the target it tests against, so every assertion here
is a p-value at a fixed seed against a target that :mod:`tests.oracles` computes by brute force,
and every one of them is paired with a negative control: the same sample checked against a
deliberately wrong target that a defect would have produced.  The control has to fail for the
assertion to mean anything.
"""
