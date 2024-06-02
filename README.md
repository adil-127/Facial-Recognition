# Anti-Spoofing and Face Recognition Integration
I have integrated an anti-spoofing model with a face recognition model to accurately identify and classify real persons versus fake persons (images or videos). The system combines the strengths of both models to enhance security measures in scenarios where facial recognition is employed.

## Features:

**Anti-Spoofing Model:** Utilizes advanced techniques to detect spoof attacks such as images(printed or any static image), masks, or videos.

**Face Recognition Model:** Identifies and classifies individuals based on facial features and patterns.

**Integration:** Seamlessly combines the anti-spoofing and face recognition models to provide robust authentication.

**Real-time Processing:** Capable of processing both images and video streams in real-time for swift decision-making.

**Api Integration:** Api are included for both adding a new person's features and for face recognition

## Benefits:
Can be easily deployed and integrated to any aplication


## Flutter Mobile App:
I have also created a Flutter Mobile app for the User Interface.

![1](https://github.com/adil-127/Facial-Recognition/assets/107793520/45176cd2-758c-4193-94a4-53e9d58b49f0)


![2](https://github.com/adil-127/Facial-Recognition/assets/107793520/eeeb7058-a480-402e-9173-81c04b34228e)

![3](https://github.com/adil-127/Facial-Recognition/assets/107793520/e7272c3b-f97a-42f2-a5b0-5e9bde3be835)

![4](https://github.com/adil-127/Facial-Recognition/assets/107793520/cc807e5e-f954-47b8-8639-4fc017a7d77b)

![5](https://github.com/adil-127/Facial-Recognition/assets/107793520/614f3c53-cae5-44d0-8320-a49fb3c20449)

![6](https://github.com/adil-127/Facial-Recognition/assets/107793520/1c5d3047-f393-4c7f-b25b-74ae500b4339)



## API Reference

#### Face Recognition

```http
  /process-frame/
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `frame` | `List` | **Required**. Frames should base64 encoded |

#### Registering facial features

```http
   /Register_User/
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `frame`      | `List` | **Required**. Frames should base64 encoded |
| `name`      | `str` | **Required**.  |



## Tech Stack

**Client:** Flutter

**Server:** Python, Fast Api


## Acknowledgements

 - [Face Recognition](https://github.com/vectornguyen76/face-recognition/tree/master)
 - [AntiSpoofing](https://github.com/computervisioneng/Silent-Face-Anti-Spoofing)
