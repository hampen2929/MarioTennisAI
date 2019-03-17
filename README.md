# MarioTennisAI
![test4](https://user-images.githubusercontent.com/34574033/54477224-3cb7c600-4849-11e9-8ce6-5398fbefddf7.gif)

We generated a model of Mario tennis using a reinforcement learning algorithm called DQN.
Mario runs on a reinforcement learning model, eventually grew to win Luigi.

## Requirements
Tested on as below.

*OS:Windows10
CPU:core i5-2400

GPU:GTX-1060

Memory:16GB

1. The dolphin emulator. You can download from [here](https://ja.dolphin-emu.org/download/). Stable version is better.
2. The Mario Power Tennis iso image.
3. Python 3. On Windows, you can use [Anaconda](https://www.anaconda.com/distribution/).

## Train


1. Launch MarioTennis on dolphin emulator.
2. Set windows looks lile the image.
![work_image](https://user-images.githubusercontent.com/34574033/54476268-9ebefe00-483e-11e9-97db-3bc0995942ab.PNG)
3. Run the proggram.

```bash
cd ./src
python Tennis_AI_normal
```

## Evaluation
![reward_point_game](https://user-images.githubusercontent.com/34574033/54472480-5d155f80-480c-11e9-906f-19879729b887.jpg)

You can see that as the number of games increases, the rewards for serving games also increase.
However, in return games, the rewards are decreasing rather than increasing.

![reward_pie_-768x388](https://user-images.githubusercontent.com/34574033/54476201-f7da6200-483d-11e9-957a-4ae5e38fa666.png)

The breakdown of points is as above. 

## Play
Inference mode hasn't implemented, yet.

## Movie
[![youtube02](https://user-images.githubusercontent.com/34574033/54472565-7539ae80-480d-11e9-8f79-593c895ac683.PNG)
](https://www.youtube.com/watch?v=OyL9Ys0tztc)

Commentary videos are uploaded to Youtube.

## Article
http://hampen2929.com/mario-tenni-ai/825

This article explains in more detail in Japanese.
