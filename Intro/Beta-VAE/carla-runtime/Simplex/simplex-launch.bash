trap "exit" INT TERM ERR
trap "kill 0" EXIT
python3 detector.py &
python3 detector1.py &
sleep 50
python2 LEC.py &
python2 perception-simplex-change.py &
python2 SS.py &
python3 liveplotter-simplex-change.py &
python2 DM.py
wait
