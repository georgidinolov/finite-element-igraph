cc_library(
	name = "basis-types",
	srcs = ["BasisTypes.cpp"],
	hdrs = ["BasisTypes.hpp"],
	deps = ["//src/igraph-0.7.1:igraph",
	        "//src/multivariate-normal:multivariate-normal"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas"],
	visibility = ["//visibility:public"],
)

cc_binary(
	name = "basis-types-test",
	srcs = ["basis-types-test.cpp"],
	includes = ["BasisTypes.hpp"],
	deps = [":basis-types"],
	copts = ["-Isrc/igraph-0.7.1/include"],	
)

cc_library(
	name = "multivariate-normal",
	srcs = ["MultivariateNormal.cpp"],
	hdrs = ["MultivariateNormal.hpp"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas"],
	visibility = ["//visibility:public"]
)

cc_binary(
	name = "multivariate-normal-test",
	srcs = ["multivariate-normal-test.cpp"],
	includes = ["MultivariateNormal.hpp"],
	deps = [":multivariate-normal"],
)