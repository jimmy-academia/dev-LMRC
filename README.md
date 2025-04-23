# development of LMRC
> goal: organize and retrieve items with LLM agent

## usage: `python -m bottom_up.main`

## log

- [2025-4-23] `improve` branch is bottom-up one shot; `master` branch will do bottom-up multi-step.
- [2025-4-21] mid_sample run finished; checked; reran file tree 200 ... good
    ==> we will check summary in file tree separately
- [2025-4-19] Enforcing summary (`python -m data.mid_sample`) => todo; complete `formatted prompt` with `item['summary']` in `main.py`
- [2025-4-18] Continue to work; mid-sample done; => 200 items run; next thing todo: tree management!!!
- [2025-4-16] New sample method, save to `improve` branch
- [2025-4-15] Bottom-up is completed; to work on improvements
    (improve branch =>> todo: work on improvement!!)
- [2025-4-11] The goal is to finish two implementations and get some results!
    - Complete bottom-up first
    - Run `python -m bottom_up.main`: from script import ... or from .sub import ... with `dev-LMRC/script.py` and `dev-LMRC/bottom_up/sub.py`

- [2025-4-7] <init> indexing batch... runnable but need rework!
- [2025-4-3] test run;

- [2025-4-2] 

working on recurssive agent `agent/recursive.py`
check `agent/idea*`
TODO: llm call functions x3, process functions x2