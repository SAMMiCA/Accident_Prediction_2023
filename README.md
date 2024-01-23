# Accident Prediction


0. Requirements
    * Python 3.6+
    * Ubuntu 20.04


1. ckpt 다운로드

아래 링크에서 파일을 다운받아 src/ckpt폴더에 둡니다.

[Onedrive Download](https://kaistackr-my.sharepoint.com/:f:/g/personal/jihui_kaist_ac_kr/EhstzIDcWVpFjeLvU1A2FLkB-3o73V_0RseJ9jz5at9jqQ?e=uqm8O9)

2. CARLA 서버 실행

CARLA 서버를 실행합니다.

```
    $ ./CarlaUE4.sh
```


3. CARLA ROS BRIDGE의 manual control 실행

carla ros bridge의 carla_ros_bridge_with_example_ego_vehicle.launch을 실행합니다.

```
    $ roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch
```


4. accident_prediction 패키지를 넣고 빌드합니다.

```
    $ roscd
    $ catkin_make
```


5. accident_prediction 패키지의 subpub.py 파일을 실행합니다. "/carla/ego_vehicle/rgb_front/image" 토픽을 받아서 50 프레임의 이미지 어레이를 메시지로 담아 img2stream 노드에서 "image_stream" 토픽을 생성합니다. 

```
    $ roscd accident_prediction/src
    $ python subpub.py
```


6. accident_prediction 패키지의 rtfm_stream.py 파일을 실행합니다. Float32MultiArray 메시지로 5개의 logit을 담아 전달합니다.

```
    $ roscd accident_prediction/src
    $ python rtfm_stream.py
```
