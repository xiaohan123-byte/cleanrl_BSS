# v2 discrete MPC migration archive

Created before the approved no-terminal-value rewrite on 2026-09-12.
The original src, tests, data_generation_test, data_generation_rl,
data_generation_optim, RL, docs, run_mpc.py and environment.yml were copied here.
Historical source files were removed from v2 only after matching each original
source file with its archived SHA256 (generated __pycache__ excluded).
Old tmp and texput.log artifacts were moved here intact.

paper_hashes.json records the pre-implementation hashes of paper_v2 files.
The paper itself and editor settings were not modified or moved by this task.
initial_git_status.txt records pre-existing user changes, including changes in
the older ExpresswayBSSs tree; those changes were not modified by this task.

The new runnable implementation is in ExpresswayBSSs_v2/. No legacy Python
module or archived input is required to run it. This archive is recovery and
research material, not a second active implementation.

Concurrent updates to 03_model.tex, appendix_linearization.tex and compiled
paper artifacts were detected after the initial snapshot (17:02 on 2026-09-12).
They were preserved; a whole-paper unchanged-hash claim is not applicable.
