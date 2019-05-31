require(ggplot2)
require(data.table)
require("gridExtra")
args = commandArgs(trailingOnly = TRUE)


list_of_files = list("profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.70-sigma_x_basis-0.20-sigma_y_basis-0.06-dx_likelihood-0.03125.R",
		     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.70-sigma_x_basis-0.20-sigma_y_basis-0.08-dx_likelihood-0.03125.R",
		     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.70-sigma_x_basis-0.20-sigma_y_basis-0.10-dx_likelihood-0.03125.R",
		     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.70-sigma_x_basis-0.25-sigma_y_basis-0.06-dx_likelihood-0.03125.R",
		     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.70-sigma_x_basis-0.25-sigma_y_basis-0.08-dx_likelihood-0.03125.R",
		     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.70-sigma_x_basis-0.25-sigma_y_basis-0.10-dx_likelihood-0.03125.R",
		     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.60-sigma_x_basis-0.25-sigma_y_basis-0.06-dx_likelihood-0.03125.R",
	      	     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.60-sigma_x_basis-0.25-sigma_y_basis-0.08-dx_likelihood-0.03125.R",
		     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.60-sigma_x_basis-0.25-sigma_y_basis-0.10-dx_likelihood-0.03125.R",
		     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.60-sigma_x_basis-0.20-sigma_y_basis-0.06-dx_likelihood-0.03125.R",
	      	     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.60-sigma_x_basis-0.20-sigma_y_basis-0.08-dx_likelihood-0.03125.R",
		     "profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.60-sigma_x_basis-0.20-sigma_y_basis-0.10-dx_likelihood-0.03125.R")

rhos = c(rep(0.70, 6), rep(0.60, 6))
sigmas.x = c(0.20, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.20, 0.20, 0.20)
sigmas.y = c(0.06, 0.08, 0.10, 0.06, 0.08, 0.10, 0.06, 0.08, 0.10, 0.06, 0.08, 0.10)
J = 16

ts_list = vector(mode="list", length=length(list_of_files))
lls_list = vector(mode="list", length=length(list_of_files))
lls_small_t_list = vector(mode="list", length=length(list_of_files))
lls_analytic_list = vector(mode="list", length=length(list_of_files))

plot.results = data.table()


for (i in seq(1,length(list_of_files))) {
    source(list_of_files[[i]])

    ts_list[[i]] = ts
    lls_list[[i]] = lls
    lls_small_t_list[[i]] = lls_small_t
    lls_analytic_list[[i]] = lls_analytic_list

    for (j in seq(1,J)) {
    	current.table = data.table(t=ts_list[[i]][[j]])
	current.table[, lls := lls_list[[i]][[j]] ]
	current.table[, lls_small_t := lls_small_t_list[[i]][[j]] ]
	current.table[, resolution := paste0("rho=", rhos[i], ", sigma_x=", sigmas.x[i], ", sigma_y=", sigmas.y[i]) ]
	current.table[, data_point := j ]

        plot.results = rbind(plot.results, current.table)
    }
}

plot_data_point <- function (j_point, data) {
	g = ggplot(plot.results[data_point==j_point,],
     	       aes(y=lls, x=t, group=resolution)) +
     	geom_line(aes(linetype=resolution,color=resolution)) +
	xlab(expression(tilde(t))) +
	ylab("log likelihood")
	if (j_point != 4) {
	   g = g + theme(legend.position="none")
	} else {
	   g = g
	}
	return (g)
}

plot.list = lapply(X=seq(1,J), FUN=plot_data_point, data=plot.results)

for (ii in seq(1,length(plot.list))) {
    print(paste0(args[1], "data-point-", ii, ".pdf"))
    ggsave(filename=paste0(args[1], "data-point-", ii, ".pdf"),
            plot=plot.list[[ii]], width = 8, height=6, units="in")
}

## pdf(paste0(args[1], "all.pdf"), 6,6)
## par(mfrow=c(ceiling(sqrt(J)), ceiling(sqrt(J))), mar=c(2,2,0,1))
## for (j in seq(1,J)) {
##     for (i in seq(1,length(list_of_files))) {
##     	if (i==1) {
## 	   plot(ts_list[[i]][[j]], lls_list[[i]][[j]], col=i, type="l", lty=i)
## 	   # lines(ts_list[[i]][[j]], lls_analytic_list[[i]][[j]], col=2, lwd=2)
## 	} else {
## 	   lines(ts_list[[i]][[j]], lls_list[[i]][[j]], col=i, lty=i)
## 	}
##     }
## }
## dev.off()


