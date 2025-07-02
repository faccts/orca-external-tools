#!/bin/bash

# Direct execution of standalone wrapper script
echo "These tests might take a few minutes."
echo "To check the progress, please see the respective outputs."
cmd="../../uma.sh HF_exttool.inp > HF_exttool.out"
echo "Test command: ${cmd}"
eval $cmd

# Execution of standalone wrapper script via ORCA optimization
cmd="$(which orca) HF_orca_ext.inp > HF_orca_ext.out"
echo "Test command: ${cmd}"
eval $cmd

# Server/client test via ORCA
# - function that kills the server on exit
killserver(){
  cmd="killall umaserver"
  echo "Stopping server: ${cmd}"
  eval $cmd
}
trap "killserver; exit" INT TERM EXIT
# - start the server
sf=HF_orca_extclient.serverout
cmd="../../umaserver.sh > $sf 2>&1 &"
echo "Starting server: ${cmd}"
eval $cmd
# - initialize the output file
of=HF_orca_extclient.out
> $of
# - wait for the server to start
while [ -z "$(grep INFO:waitress:Serving $sf)" ]; do echo "Waiting for server" >> $of; sleep 1s; done
# - start the ORCA job
cmd="$(which orca) HF_orca_extclient.inp >> $of"
echo "Test command: ${cmd}"
eval $cmd

# stop the server between tests
killall umaserver

# Parallel server/client GOAT test
# - start the server (4 threads)
sf=H2O_goat_extclient.serverout
cmd="../../umaserver.sh -n 4 > $sf 2>&1 &"
echo "Starting server: ${cmd}"
eval $cmd
# - initialize the output file
of=H2O_goat_extclient.out
> $of
# - wait for the server to start
while [ -z "$(grep INFO:waitress:Serving $sf)" ]; do echo "Waiting for server" >> $of; sleep 1s; done
# - start the ORCA job
cmd="$(which orca) H2O_goat_extclient.inp >> $of"
echo "Test command: ${cmd}"
eval $cmd
