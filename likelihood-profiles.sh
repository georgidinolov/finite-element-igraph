#!/bin/bash

## THIS NEEDS TO BE RUN FROM ~/PDE-solvers
cd ~/PDE-solvers

bazel --output_user_root=/home/gdinolov-tmp build //src/finite-element-igraph:likelihood-profiles

number_data_points=($4)
rho_basis=($1)
sigma_x_basis=($2)
sigma_y_basis=($3)
dx_numerical_likelihood=0.03125
target_dir=~/PDE-solvers/src/finite-element-igraph/
data_rho=($6)
data_n=($5)
prefix=profile-test-data-rho-${data_rho}-n-${data_n}-dx-250-
data_file=./src/kernel-expansion/documentation/chapter-2/data/mle-data-sets-rho-${data_rho}-n-${data_n}/BM-data-set-0.csv

./bazel-bin/src/finite-element-igraph/likelihood-profiles ${number_data_points} ${rho_basis} ${sigma_x_basis} ${sigma_y_basis} ${dx_numerical_likelihood} ${target_dir}${prefix} ${data_file}
