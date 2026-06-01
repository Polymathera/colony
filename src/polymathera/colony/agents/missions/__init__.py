"""First-party mission packages shipped by ``polymathera-colony``.

Each sub-package defines a coordinator (and optionally a worker) for
one ``MissionSpec`` registered in
:data:`polymathera.colony.agents.configs._BUILTIN_MISSIONS`.

Code-analysis sample missions still live under
``colony/samples/code_analysis/`` — those predate this package and stay
in place; new core missions land here so their FQN is stable and
short.
"""
