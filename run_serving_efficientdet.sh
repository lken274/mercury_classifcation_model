sudo systemctl daemon-reload
sudo systemctl restart docker
sudo docker run --runtime=nvidia -p 8500:8500 -p 8501:8501 --rm --mount type=bind,source=/home/logan/Desktop/tf_models/blemish_detector/inference_graph/saved_model/efficientdet,target=/models/blemish_detector --mount type=bind,source=/home/logan/Desktop/tf_models/blemish_detector/docker_config/,target=/docker_config -e MODEL_NAME=blemish_detector -t tensorflow/serving:nightly-gpu --enable_batching --batching_parameters_file="/docker_config/batch_params"
