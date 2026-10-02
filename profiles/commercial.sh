# Drops DFlash2, which is CC BY-NC-ND 4.0 and so non-commercial only, and drafts with the
# checkpoint's own MTP head instead. apex-flash-1 is MIT and TensorFold is Apache-2.0, so
# nothing in this profile restricts commercial use.
#
# The cost is concurrency: the batched verify window in patches 0026-0030 handles several
# streams only when DFlash2 drafts them, so PARALLEL drops to 1. One request at a time,
# somewhat slower per token. See the drafter table in README.md.
#
#   . profiles/commercial.sh && ./start.sh restart
export DRAFTER=mtp
export PARALLEL=1
export CONTEXT=262144
export MEMORY_RESERVE_GIB=24
export VISION=0
