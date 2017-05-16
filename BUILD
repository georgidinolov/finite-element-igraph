cc_library(
	name = "bivariate-solver",
	srcs = ["BivariateSolver.cpp"],
	hdrs = ["BivariateSolver.hpp"],
	deps = ["//src/igraph-0.7.1:igraph",
		":basis-types"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal",
		 "-fopenmp"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas",
		    "-fopenmp"],	
	visibility = ["//visibility:public"],
)

cc_library(
	name = "basis-types",
	srcs = ["BasisTypes.cpp"],
	hdrs = ["BasisTypes.hpp"],
	deps = ["//src/igraph-0.7.1:igraph",
		":basis-element-types"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal",
		 "-lm"],
	visibility = ["//visibility:public"],
)

cc_library(
	name = "basis-element-types",
	srcs = ["BasisElementTypes.cpp",
	        "LinearCombinationElement.cpp",
	        "BivariateSolverClassical.cpp"],
	hdrs = ["BasisElementTypes.hpp"],
	deps = ["//src/igraph-0.7.1:igraph",
	        "//src/multivariate-normal:multivariate-normal"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal",
		 "-lm",
		 "-lgsl",
		 "-lgslcblas",
		 "-O3"],
	visibility = ["//visibility:public"],
)

cc_library(
	name = "rmath",
	hdrs = ["Rmath.h"],
	copts = ["-lRmath", "-lm", "-lcmath"],
)

cc_binary(
	name = "bivariate-solver-test",
	srcs = ["bivariate-solver-test.cpp"],
	includes = ["BivariateSolver.hpp"],
	deps = [":bivariate-solver",
    	        "//src/brownian-motion:2d-brownian-motion",
		"//src/images-expansion:2d-advection-diffusion-images"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/brownian-motion",
	      	 "-Isrc/multivariate-normal",
		 "-Isrc/images-expansion"],
	visibility = ["//visibility:public"],
)

cc_binary(
	name = "bivariate-solver-test-parallel",
	srcs = ["bivariate-solver-test-parallel.cpp"],
	includes = ["BivariateSolver.hpp"],
	deps = [":bivariate-solver",
    	        "//src/brownian-motion:2d-brownian-motion"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/brownian-motion",
	      	 "-Isrc/multivariate-normal",
		 "-fopenmp",
		 "-O3"],
	linkopts = ["-fopenmp", "-lm"],
	visibility = ["//visibility:public"],
)


cc_binary(
	name = "bivariate-solver-classical-test",
	srcs = ["bivariate-solver-classical-test.cpp"],
	includes = ["BasisElementTypes.hpp"],
	deps = [":basis-element-types"],
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
	      	 "-Isrc/multivariate-normal",
		 "-O"],
)

cc_binary(
	name = "trig-interpolant-test",
	srcs = ["trig-interpolant-test.cpp"],
	includes = ["BasisElementsTypes.hpp"],
	deps = [":basis-element-types"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal",
		 "-O"],
)