# Provenance of the files added for the revision (release/v2.0, work in progress)

Every file listed below is a byte-for-byte copy of a file in the workspace where the revised results were produced. The sha256 of each copy was compared with its source at copy time.

| Source repository | Commit |
|---|---|
| `hp_vpinn` (branch `jax_inverse_problem_hpVPINN_grad_balance`) | `7cc4e4a` |
| workspace root (scripts and data at the top level) | `0b18f97` |

Layout: workspace root files are at the top level of this repository, files from `hp_vpinn/` are under `hp_vpinn/`, and the cluster launchers are under `slurm/`. The scripts compute paths from their own location (`REPO_ROOT`), so the `.py` files must stay where they are. The launchers in `slurm/` were copied unchanged: they contain `cd ~/manuscript_review_runs` and expect to be submitted from `slurm/`.

The layout of the original submission (`src/`, `experiments/`, `postprocessing/`, `figures/`, `results/`, `data/`) is still present on this branch and is not used by the revised scripts; it remains available in the tag `v1.0-submission`.

## Not included on purpose

- The collocation PINN (frontx) code, results and final-run drivers: available upon reasonable request. The `--model frontx` backends and the conventional-inversion scripts require the `frontx` package.
- Scripts of the superseded experimental Brooks and Corey variant, the Stage B (replication) tooling, and workspace utilities that only serve the development workspace.
- Generators of main-text figures 4-6 and of the bounds-test figure.

## Known cross-references to files that are not in this repository

- `hp_vpinn/final_run_common.py` (docstring) mentions `brooks_corey.py`.
- `sensitivity_noise_density.py` (comments) refers to `hparam_confirm_top1.py` as the origin of its ranking criterion.

## sha256 of the copied files

```
80f17afb19b2a0dda44e2356b487cffd43fc3b7f3ceb37e6d1214196d52fd8f7 hp_vpinn/ constitutive_model.py
7b512d127ff6e287dc73b7ac6cfbfdbff550d135c5b489ec4cfbd9e3e7b3a72b hp_vpinn/ datasets.py
73f0af043590ee2a61b6446918b3f534be38d5c4dc7e6db1c414bb1db34acb5b hp_vpinn/ nn_model.py
84bf04dc33f7250b213af73d187ed3192c3f831b11f1725dece911183709e1c5 hp_vpinn/ pde.py
9bd568049a54ca710eab3e3bfdd1598706d9d8fd90faec84f8b5d1ffabc2ab54 hp_vpinn/ pinn_model.py
254a361590a9375a8d2cd7a3cc42e536230e8fd94d773f88a916aa92f16c466c hp_vpinn/ plot_utils.py
8af57a9158635135c0df0a8bf749b94cb92b13c37ecab377ec35fbe8d20f7227 hp_vpinn/ strong_form_residual.py
09bf241d54ba50fb087d2e0c79ae7d9ac06ac0a2c5687c19b30c763ffa1da73c hp_vpinn/ train_vpinn.py
17fe6526b53f163a32921a6ebb57bddf3218b09ec98e9d37e7bbc87f397fdccd hp_vpinn/ transformations.py
826e287c2b1dca0fd767c36e9b792fd1eddeaed5aa29ab479ec30804b113bab9 hp_vpinn/ vpinn_loss.py
f888608a48f01b32f3a90246a000f6f219c6b1d2c8e3e4e1f0bcddb4119f4252 hp_vpinn/ pyproject.toml
c043752d7cb26f3a730c5aa67a58e34d27204495c7d95958368e82b1e5780d7b hp_vpinn/ uv.lock
d7a55add8e53f98287066bd3546bc971b7b067a7546c4f26096de8d9253d7667 hp_vpinn/ .gitignore
8aad3411632490a1b97b7fdcd638f484d845af66c270fcd04a1d99e56c852364 hp_vpinn/ final_run_common.py
84e6b9d7fd250062fbc5a161c428bd1c0fe85b5a758e603c6ecd6f7980d8417d hp_vpinn/ grenoblesand.py
ff19b0a9dd90f344303b4905d34fdedc65588cc76135e37f3b2915410347b37d hp_vpinn/ letd.py
60394ccd0415164ee4000d4ecc6135ccda26f458a758dbdc2e8d3022af8145db hp_vpinn/ vangenuchten.py
3d0e6f8dc799be8e184263bbe16897203da5a8b1d92b1f7de9ddf08fbc5af43b hp_vpinn/ profile_figures.py
ff6fc922f95f6097f2f4088333598b420d97a8e8fac646156991c8f2bb552779 hp_vpinn/ combined_profiles.py
7f926735670345002a99635e5f407a87655be0e81a5a5c7ef144808218bb95e6 hp_vpinn/ early_stopping_analysis.py
4f274807b2e4cc1839a19becde37236f47174068d554bd75fc1c0491b97c9416 hp_vpinn/ multistart.py
b3cd3a895de1a4eb0ed0db96fc3302a76bd6ae46df4bfcc835fe8aec0233282e  data_grenoblesand.dat
e8d7f88aba24181e884f1a342a4c56b7b41c53acb02f725b4bef900f18f5faf8  data_letd.dat
e8d7f88aba24181e884f1a342a4c56b7b41c53acb02f725b4bef900f18f5faf8  data_vangenuchten.dat
baef76fe1808e7763edb37ce48856814d5c218816e6bee3d257fdb1f43aa1ebb  data_brooks_corey_new.dat
9056a13e4340b7a4bbc214cccc3dd27148d558800e17dc4a7efa25deae7003ef  hparam_search_grenoblesand.py
a61b9dce9f4be0e0e620ec588f26dbf3b4313a0c74dcad40497822c02d5db80d  hparam_search_letd.py
c7b0c215cbf3e9bd9ca34c0bf2a5d5b60a922a1671d54ec0d9efa6dd8283cb6c  hparam_search_vangenuchten.py
0ee3adf8820d7d7c893b9db0bcf16d47b0c590c40dfba7fffbd7e3eaeb32822a  hparam_search_brooks_corey_new.py
920510da002b2439e7b3784c49f337b4ac4adb68f4277a20c8f829c970dc0f51  hparam_postprocess.py
b25ec4d72a56ade0856f961ad0e37141d44dabbfd509b8e3111f4660ee8bf4f8  quadrature_check_grenoblesand.py
0a01014a2f45fb108030183f96d37131e82a1295e3b77393ebdd0ab2b15e5cc3  sensitivity_noise_density.py
0978f0b6a632c7c26fa60706ce7f20900aad00b5e33f8ca7b9980ec22b5f05cf  postprocess_sensitivity.py
34b75a20e4e0aa5b2369c4a8f47632e55db308280893ea97ee04c2b383253302  conventional_inverse.py
6e7001a92bf6a3e1a89ad26e60e8d53706b1dd313d3e8a58cb9e013d07643869  benchmark_conventional.py
33e9759a737a7a597c54104af680a100b0d3c06e02a29d7e6b3f788cd719d944  final_runs_to_forward_check.py
efabb3fb7e9ba50aff01f09289801594e89e6d8bd2141c3a87641b2d868b7388  rerun_forward_check_tight.sh
415ca7510dcb7a3d80e0fb9836dc77e7090aa5222c993b4ad46a2e52973462b8  slurm/run_hparam_grenoblesand.sh
80f8834f86f38a39c36eeb4c43258abe6db42ffa31c229d9f328d96bf3cdd42e  slurm/run_hparam_letd.sh
5094f70a509e4042f2274c9121604da20a9e1b044a20aaebe859df9964fefacf  slurm/run_hparam_vangenuchten.sh
abbcf4d056ca474edf0ec04d0d2cc45f5f9497de2c056b48ab27d270d06cb0bf  slurm/submit_all_grenoblesand.sh
44f5b18f497b4e2bf5da10a404a1ff2593a78f9d5f6dd9e598ae856f9348a585  slurm/submit_all_letd.sh
0d27a416cfaff6d88358ae9bb0a46c565a919ad2bb9f4a8bc8fb0d2d2f00d7a2  slurm/submit_all_vangenuchten.sh
6b187b643cdbb080ba46d2dad0029a9e6a920f0531ceccd342672aaf76e914f6  slurm/submit_all_brooks_corey_new.sh
f2a858188dfe0758a53217bc463940a1d6b27f663064d5a5c0ff5cced313b412  slurm/slurm_run_hparam_grenoblesand.sh
7027d26c50e8adbe20c3fc85b47166c487d50dc18b5b69152b054885aaacc3fc  slurm/slurm_run_hparam_letd.sh
06268c0576e7eb8dad67e8b8c1a83c51998693a1f0a815af0ecd8f6727fbfe10  slurm/slurm_run_hparam_vangenuchten.sh
785510b6cdd35d6cc3c83c65bf02d6bb3797c3a2e1ed7169db06c2861494965f  slurm/slurm_run_hparam_brooks_corey_new.sh
29fc1a2d2bb56af84d0d3957300bca7cebae6bfd5ee0afef6a05bda00a3303ce  slurm/slurm_run_multistart.sh
f8aa498c07988525bb903be0465963f988cb080a9af01e464f418655acabfeba  slurm/slurm_run_sensitivity.sh
a3ea0978872a74fe3bfc0da56bf7fb15527274d1f204daa1453b2cefe11f4d1f  slurm/submit_sensitivity.sh
```
