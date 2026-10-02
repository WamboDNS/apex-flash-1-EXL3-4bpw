# Default profile. Already the defaults in scripts/config.sh; here so the choice is explicit
# and so `. profiles/headroom.sh` undoes another profile in the same shell.
#
# Leaves roughly 24 GiB free on each Spark for other GPU work, at the cost of the window.
export CONTEXT=262144
export MEMORY_RESERVE_GIB=24
export VISION=0
