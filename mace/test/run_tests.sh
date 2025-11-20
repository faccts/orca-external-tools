#!/bin/bash

# Output directory for artifacts (can override with OUTDIR=/path)
OUTDIR="${OUTDIR:-.cache}"
mkdir -p "$OUTDIR"

# Direct execution of standalone wrapper script
echo "These tests might take a few minutes."
echo "To check the progress, please see the respective outputs."
cmd="../../mace.sh HF_exttool.inp > \"$OUTDIR\"/HF_exttool.out"
echo "Test command: ${cmd}"
eval $cmd
mv -f ./*.engrad "$OUTDIR" 2>/dev/null || true

# Check if ORCA is available; if not, skip ORCA-based tests
ORCA_CMD=""
if [ -n "${ORCA:-}" ] && [ -x "${ORCA}" ]; then
  ORCA_CMD="${ORCA}"
elif command -v orca >/dev/null 2>&1 ; then
  ORCA_CMD="$(command -v orca)"
fi

# Check if ORCA is available; if not, skip ORCA-based tests
if [ -z "$ORCA_CMD" ]; then
  echo "ORCA not found. Set ORCA env var or add to PATH. Skipping ORCA-based tests (orca_ext, client, GOAT)."
  exit 0
fi

# Execution of standalone wrapper script via ORCA optimization
cmd="\"$ORCA_CMD\" HF_orca_ext.inp > \"$OUTDIR\"/HF_orca_ext.out"
echo "Test command: ${cmd}"
eval $cmd

# Server/client test via ORCA
# - function that kills the server on exit
killserver(){
  cmd="killall maceserver"
  echo "Stopping server: ${cmd}"
  eval $cmd
}
trap "killserver; exit" INT TERM EXIT
# - start the server
sf="$OUTDIR/HF_orca_extclient.serverout"
cmd="../../maceserver.sh > \"$sf\" 2>&1 &"
echo "Starting server: ${cmd}"
eval $cmd
# - initialize the output file
of="$OUTDIR/HF_orca_extclient.out"
> "$of"
# - wait for the server to start
WAITED=0
while [ -z "$(grep -E "Serving|Running on" "$sf")" ]; do echo "Waiting for server" >> "$of"; sleep 1s; WAITED=$((WAITED+1)); if [ $WAITED -gt 30 ]; then echo "Timeout waiting for server" >> "$of"; break; fi; done
# - start the ORCA job
cmd="\"$ORCA_CMD\" HF_orca_extclient.inp >> \"$of\""
echo "Test command: ${cmd}"
eval $cmd

# stop the server between tests
killall maceserver || true

# Parallel server/client GOAT test
# - start the server (4 threads)
sf="$OUTDIR/H2O_goat_extclient.serverout"
cmd="../../maceserver.sh -n 4 > \"$sf\" 2>&1 &"
echo "Starting server: ${cmd}"
eval $cmd
# - initialize the output file
of="$OUTDIR/H2O_goat_extclient.out"
> "$of"
# - wait for the server to start
WAITED=0
while [ -z "$(grep -E "Serving|Running on" "$sf")" ]; do echo "Waiting for server" >> "$of"; sleep 1s; WAITED=$((WAITED+1)); if [ $WAITED -gt 30 ]; then echo "Timeout waiting for server" >> "$of"; break; fi; done
# - start the ORCA job
cmd="\"$ORCA_CMD\" H2O_goat_extclient.inp >> \"$of\""
echo "Test command: ${cmd}"
eval $cmd
