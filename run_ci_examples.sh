set -e

pushd examples/skorch || exit 1
ray stop || true
echo "================"
echo "Running examples"
echo "================"
echo "running basic_example.py" && python basic_example.py
echo "running callbacks_example.py" && python callbacks_example.py
echo "running datasets_example.py" && python datasets_example.py
echo "running datasets_pipeline_example.py" && python datasets_pipeline_example.py
echo "running prediction_example.py" && python prediction_example.py
echo "running skorch_datasets_example.py" && python skorch_datasets_example.py
popd