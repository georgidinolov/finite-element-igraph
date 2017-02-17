cc_library(
	name = "basis-types",
	srcs = ["BasisTypes.cpp"],
	hdrs = ["BasisTypes.hpp"],
	visibility = ["//visibility:public"],
	deps = ["//src/igraph-0.7.1:igraph"],
	copts = ["-Isrc/igraph-0.7.1/include"],
)

cc_binary(
	name = "basis-types-test",
	srcs = ["basis-types-test.cpp"],
	includes = ["BasisTypes.hpp"],
	deps = [":basis-types"],
	copts = ["-Isrc/igraph-0.7.1/include"],	
)