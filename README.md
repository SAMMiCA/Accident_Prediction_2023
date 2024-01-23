# Accident Prediction

RTFM 모델을 이용해 이미지 스트림으로부터 사고 확률을 계산하는 코드입니다.


0. Requirements
    * Python 3.6+
    * Ubuntu 20.04


1. CARLA 서버 실행

CARLA 서버를 실행합니다.

```
    $ ./CarlaUE4.sh
```


2. CARLA ROS BRIDGE의 manual control 실행

carla ros bridge의 carla_ros_bridge_with_example_ego_vehicle.launch을 실행합니다.

```
    $ roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch
```


3. accident_prediction 패키지를 넣고 빌드합니다.

```
    $ roscd
    $ catkin_make
```


4. accident_prediction 패키지의 subpub.py 파일을 실행합니다. "/carla/ego_vehicle/rgb_front/image" 토픽을 받아서 50 프레임의 이미지 어레이를 메시지로 담아 img2stream 노드에서 "image_stream" 토픽을 생성합니다. 

```
    $ roscd accident_prediction/src
    $ python subpub.py
```


5. accident_prediction 패키지의 rtfm_stream.py 파일을 실행합니다. Float32MultiArray 메시지로 5개의 logit을 담아 전달합니다.

```
    $ roscd accident_prediction/src
    $ python rtfm_stream.py
```