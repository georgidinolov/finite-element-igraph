cc_library(
	name = "bivariate-solver-classical",
	srcs = ["BivariateSolverClassical.cpp"],
	hdrs = ["BivariateSolverClassical.hpp"],
	deps = ["//src/igraph-0.7.1:igraph",
		":basis-element-types"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas"],	
)

cc_library(
	name = "basis-types",
	srcs = ["BasisTypes.cpp"],
	hdrs = ["BasisTypes.hpp"],
	deps = ["//src/igraph-0.7.1:igraph",
		":basis-element-types"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal"],
	visibility = ["//visibility:public"],
)

cc_library(
	name = "basis-element-types",
	srcs = ["BasisElementTypes.cpp"],
	hdrs = ["BasisElementTypes.hpp", "BasisTypes.hpp"],
	deps = ["//src/igraph-0.7.1:igraph",
	        "//src/multivariate-normal:multivariate-normal"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal"],
	visibility = ["//visibility:public"],
)

cc_binary(
	name = "bivariate-solver-classical-test",
	srcs = ["bivariate-solver-classical-test.cpp"],
	includes = ["BivariateSolverClassical.hpp"],
	deps = [":bivariate-solver-classical"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal"],
	visibility = ["//visibility:public"],
)

cc_binary(
	name = "basis-types-test",
	srcs = ["basis-types-test.cpp"],
	includes = ["BasisTypes.hpp"],
	deps = [":basis-types"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal"],
)

cc_binary(
	name = "element-types-test",
	srcs = ["element-types-test.cpp"],
	includes = ["BasisElementsTypes.hpp"],
	deps = [":basis-element-types"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal"],
)