# Upstream's settings: the checkpoint's full window and the vision tower, with only the
# 4.5 GiB or so of free memory a Spark is left with. Use this when the Sparks are doing
# nothing else.
#
#   . profiles/max-context.sh && ./start.sh restart
export CONTEXT=1048576
export MEMORY_RESERVE_GIB=14.5
export VISION=1
