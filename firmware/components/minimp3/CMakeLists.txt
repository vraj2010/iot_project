idf_component_register(SRCS "minimp3_impl.c" INCLUDE_DIRS ".")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-function -Wno-maybe-uninitialized)
