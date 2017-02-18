cc_library(
	name = "basis-types",
	srcs = ["BasisTypes.cpp"],
	hdrs = ["BasisTypes.hpp"],
	deps = ["//src/igraph-0.7.1:igraph",
	        "//src/multivariate-normal:multivariate-normal"],
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
	includes = ["BasisTypes.hpp"],
	deps = [":basis-types"],
	copts = ["-Isrc/igraph-0.7.1/include",
	      	 "-Isrc/multivariate-normal"],
)