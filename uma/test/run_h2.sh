f='h2_extopt'
PROCS=$(grep "nprocs" $f.inp | sed "s/^.*procs *//" | awk '{print $1}')

ORCA=$(which orca)

../../umaserver.sh & # run an initial instance which will create a default host 127.0.0.1:8888

so="h2_extopt.serverout"
> $so
../../umaserver.sh -n $PROCS > $so 2>&1 &

while true; do
  if grep -q "INFO:waitress:Serving" $so; then
    export UMA_BIND=$(grep "INFO:waitress:Serving" $so | awk '{print $NF}' | sed "s/http:\/\///")
    echo "Started server on port $UMA_BIND"
    export UMASERVER_PID=$(grep "UMASERVER_PID:" $so | awk '{print $NF}')
    echo "umaserver running with PID: $UMASERVER_PID"
    break
  fi
  #echo "Waiting for server startup"
  sleep 1
  elapsed=$((elapsed +1))
  if [ $elapsed -ge "10" ] ; then
    echo "Timeout"
    exit 1
  fi
done

mkdir tmp
cp $f.inp tmp
cd tmp
echo "Starting calculation"
$ORCA $f.inp > ../output/$f.out
cd ..
rm -rdf tmp

kill $UMASERVER_PID # only kill the relevant umaserver (illustrative)
#rm $so
echo "Completed job"
pkill umaserver # kill all umaserver 

