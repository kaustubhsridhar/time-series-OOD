trap "exit" INT TERM ERR
trap "kill 0" EXIT

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg           # 0.9.6
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py2.7-linux-x86_64.egg           # 0.9.6

python3 detector.py &
python3 detector1.py &
sleep 50
python3 LEC.py &
python3 perception.py &
python3 SS.py &
#python3 liveplotter.py &
python3 DM.py
wait
