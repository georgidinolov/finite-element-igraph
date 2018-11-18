list_of_files = list(# "profile-test-data-rho-0.95-n-8-dx-250--number-points-8-rho_basis-0.60-sigma_x_basis-0.08-sigma_y_basis-0.08-dx_likelihood-0.03125.csv",
"profile-test-data-rho-0.95-n-8-dx-250--number-points-8-rho_basis-0.50-sigma_x_basis-0.08-sigma_y_basis-0.08-dx_likelihood-0.03125.csv",
# "profile-test-data-rho-0.95-n-8--number-points-8-rho_basis-0.40-sigma_x_basis-0.08-sigma_y_basis-0.08-dx_likelihood-0.03125.csv",
# "profile-test-data-rho-0.95-n-8-dx-200--number-points-8-rho_basis-0.40-sigma_x_basis-0.08-sigma_y_basis-0.08-dx_likelihood-0.03125.csv",
# "profile-test-data-rho-0.95-n-8--number-points-8-rho_basis-0.60-sigma_x_basis-0.15-sigma_y_basis-0.08-dx_likelihood-0.03125.csv",
# "profile-test-data-rho-0.95-n-8--number-points-8-rho_basis-0.60-sigma_x_basis-0.12-sigma_y_basis-0.08-dx_likelihood-0.03125.csv",
# "profile-test-data-rho-0.95-n-8--number-points-8-rho_basis-0.60-sigma_x_basis-0.10-sigma_y_basis-0.08-dx_likelihood-0.03125.csv",
# "profile-test-data-rho-0.95-n-8-dx-200--number-points-8-rho_basis-0.60-sigma_x_basis-0.08-sigma_y_basis-0.08-dx_likelihood-0.03125.csv",
"profile-test-data-rho-0.95-n-8--number-points-8-rho_basis-0.50-sigma_x_basis-0.08-sigma_y_basis-0.08-dx_likelihood-0.03125.csv")

ts_list = vector(mode="list", length=length(list_of_files))
lls_list = vector(mode="list", length=length(list_of_files))
lls_small_t_list = vector(mode="list", length=length(list_of_files))

for (i in seq(1,length(list_of_files))) {
    source(list_of_files[[i]])
    ts_list[[i]] = ts
    lls_list[[i]] = lls
    lls_small_t_list[[i]] = lls_small_t
}

J = length(lls)

par(mfrow=c(ceiling(sqrt(J)), ceiling(sqrt(J))), mar=c(1,1,1,1))
for (j in seq(1,J)) {
    for (i in seq(1,length(list_of_files))) {
    	if (i==1) {
	   plot(ts_list[[i]][[j]], lls_list[[i]][[j]], col=i, type="l")
	   lines(ts_list[[i]][[j]], lls_small_t_list[[i]][[j]], col=2, type="l", lwd=2)
	} else {
	   lines(ts_list[[i]][[j]], lls_list[[i]][[j]], col=i)
	}	
    }   
}


