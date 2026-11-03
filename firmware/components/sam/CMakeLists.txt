# SAM - Software Automatic Mouth (github.com/s-macke/SAM). Offline fallback voice.
idf_component_register(
    SRCS "sam.c" "render.c" "reciter.c" "debug.c" "sam_glue.c"
    INCLUDE_DIRS "."
)
# Vendored 1980s-style C: keep its warnings out of our build output.
target_compile_options(${COMPONENT_LIB} PRIVATE -w)
