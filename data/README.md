# Data files

Comma-separated text files without header, one row per point of the moisture-content profile.
The first column is the Boltzmann variable `o = x / sqrt(t)`, the second the moisture content `theta`.

| File | Case | Rows | Columns | Origin |
|---|---|---|---|---|
| `data_grenoblesand.dat` | 1 | 100 | `o`, `theta` | Noise-free reference curve of the van Genuchten model (`K_s = 15.37 cm/h`, `alpha = 0.0432 1/cm`, `m = 0.5096`, `l = 0.5`, `theta_s = 0.312`, `theta_r = 0`), computed once with the exact solver of [`frontx`](https://github.com/gerlero/frontx). `o` in `cm h^-1/2`. |
| `data_vangenuchten.dat` | 2 | 141 | `o`, `theta`, `std` | Experimental profile of Whatman No. 1 paper, from the data set of Gerlero et al. (2022). Same file as `data_letd.dat`. `o` in `m s^-1/2`. |
| `data_letd.dat` | 3 | 141 | `o`, `theta`, `std` | The same experimental profile, used with the LETd model. |
| `data_brooks_corey.dat` | 4 | 5000 | `o`, `theta` | Noise-free reference curve of the Brooks and Corey model for Grenoble sand, generated with porousMicroTransport (Gerlero et al. 2024, *Talanta Open* 10, 100350), which solves the complete governing PDE with the finite-volume method; the reference parameters are those reported in the manuscript. |

The data files of Cases 1 and 4 hold the noise-free curve: the noise and the train/test split are generated
by the search drivers from fixed seeds (see "Reproducibility" in the main README).

## Experimental data (Cases 2 and 3)

The profile is the data set of

> G. S. Gerlero, A. R. Valdez, R. Urteaga, P. A. Kler (2022). Validity of capillary imbibition models in
> paper-based microfluidic applications. *Transport in Porous Media* 141(2), 359-378.
> https://doi.org/10.1007/s11242-021-01724-w

which is published in the public repository https://github.com/gerlero/frontx, distributed under the Apache
License 2.0, as `src/frontx/examples/data/validity/processed.npz` (commit `e04d264`). The columns of
`data_letd.dat` and `data_vangenuchten.dat` follow from that file: `o` is its array `o`, and, with
`theta_s = 0.7` and `theta_i = 0.025`,

```
theta = theta_i + (theta_s - theta_i) * I
std   = (theta_s - theta_i) * std_I
```

where `I` and `std_I` are its arrays `I` and `std`. This was checked on the files of this repository: the three
columns are equal to the values computed from the upstream arrays, with no difference. Please cite the paper
above when using these data.

## Checksums (sha256)

```
b3cd3a895de1a4eb0ed0db96fc3302a76bd6ae46df4bfcc835fe8aec0233282e  data_grenoblesand.dat
e8d7f88aba24181e884f1a342a4c56b7b41c53acb02f725b4bef900f18f5faf8  data_vangenuchten.dat
e8d7f88aba24181e884f1a342a4c56b7b41c53acb02f725b4bef900f18f5faf8  data_letd.dat
baef76fe1808e7763edb37ce48856814d5c218816e6bee3d257fdb1f43aa1ebb  data_brooks_corey.dat
```
