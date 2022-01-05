set -e

pushd examples || exit 1
ray stop || true
echo "================"
echo "Running examples"
echo "================"
echo "running basic_example.py" && python basic_example.py --num-cpus 8
echo "running callbacks_example.py" && python callbacks_example.py --num-cpus 8
echo "running checkpoint_example.py" && python checkpoint_example.py --num-cpus 8
echo "running classification_example.py" && python classification_example.py --num-cpus 8
echo "running datasets_example.py" && python datasets_example.py --num-cpus 8
echo "running datasets_pipeline_example.py" && python datasets_pipeline_example.py --num-cpus 8
echo "running prediction_example.py" && python prediction_example.py --num-cpus 8
echo "running readme_example.py" && python readme_example.py --num-cpus 8
echo "running readme_datasets_example.py" && python readme_datasets_example.py --num-cpus 8
echo "running skorch_datasets_example.py" && python skorch_datasets_example.py --num-cpus 8
echo "running tabnet_example.py" && python tabnet_example.py --num-cpus 8
popd