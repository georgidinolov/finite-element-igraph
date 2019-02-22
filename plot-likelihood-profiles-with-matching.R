require(ggplot2)
require(data.table)
require("gridExtra")
args = commandArgs(trailingOnly = TRUE)

list_of_files = list("profile-test-data-rho-0.95-n-16-dx-300--number-points-16-rho_basis-0.60-sigma_x_basis-0.10-sigma_y_basis-0.10-dx_likelihood-0.0078125.R")

ts_list = vector(mode="list", length=length(list_of_files))
lls_list = vector(mode="list", length=length(list_of_files))
lls_small_t_list = vector(mode="list", length=length(list_of_files))
lls_analytic_list = vector(mode="list", length=length(list_of_files))
lls_matched_list = vector(mode="list", length=length(list_of_files))
lls_ansatz_list = vector(mode="list", length=length(list_of_files))
lls_LS_list = vector(mode="list", length=length(list_of_files))

plot.results = data.table()

rhos = c(0.60)
sigmas = c(0.10)

J = 16
for (i in seq(1,length(list_of_files))) {
    source(list_of_files[[i]])

    ts_list[[i]] = ts
    lls_list[[i]] = lls
    lls_small_t_list[[i]] = lls_small_t
    lls_analytic_list[[i]] = lls_analytic
    lls_matched_list[[i]] = lls_matched
    lls_ansatz_list[[i]] = lls_ansatz
    lls_LS_list[[i]] = lls_LS

    for (j in seq(1,J)) {
    	current.table = data.table(t=ts_list[[i]][[j]])
	current.table[, lls := lls_list[[i]][[j]] ]
	current.table[, type := "Galerkin"]
	current.table[, resolution := paste0("rho=", rhos[i], ", sigma=", sigmas[i]) ]
	current.table[, data_point := j ]
        plot.results = rbind(plot.results, current.table)


    	## current.table = data.table(t=ts_list[[i]][[j]])
	## current.table[, lls := lls_ansatz_list[[i]][[j]] ]
	## current.table[, type := "Ansatz"]
	## current.table[, resolution := paste0("rho=", rhos[i], ", sigma=", sigmas[i]) ]
	## current.table[, data_point := j ]
        ## plot.results = rbind(plot.results, current.table)


	current.table = data.table(t=ts_list[[i]][[j]])
	current.table[, lls := lls_matched_list[[i]][[j]] ]
	current.table[, type := "Matched"]
	current.table[, resolution := paste0("rho=", rhos[i], ", sigma=", sigmas[i]) ]
	current.table[, data_point := j ]
        plot.results = rbind(plot.results, current.table)

	current.table = data.table(t=ts_list[[i]][[j]])
	current.table[, lls := lls_LS_list[[i]][[j]] ]
	current.table[, type := "Least Squares"]
	current.table[, resolution := paste0("rho=", rhos[i], ", sigma=", sigmas[i]) ]
	current.table[, data_point := j ]
        plot.results = rbind(plot.results, current.table)
    }
}

	plot_data_point <- function (j_point, data) {
	g = ggplot(plot.results[data_point==j_point,],
     	       aes(y=lls, x=t, group=type)) +
     	geom_line(aes(linetype=type,color=type)) +
	xlab(expression(tilde(t))) +
	ylab("log likelihood") +
	ylim(-7,7)
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
            plot=plot.list[[ii]], width = 4, height=4, units="in")
}