"""The test suite.

A package rather than a plain directory, so that pytest imports every module under its full
dotted name and puts the repository root on ``sys.path`` rather than the ``tests`` directory
itself.  The search spaces the tests share, and the oracles that say what a draw from them should
look like, live here as :mod:`tests.spaces` and :mod:`tests.oracles`, and a test in any
subdirectory imports them under that name.
"""
