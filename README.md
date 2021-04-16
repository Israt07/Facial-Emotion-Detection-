# FACE MASK DETECTION USING DEEP LEARNING

## A. PROJECT SUMMARY

**Project Title:** Facial Emotion Detection using Deep Learning

**Team Members:** 
- Israt Jahan Bhuiyan
- A Bhuiyan
- B Bhuiyan
- C Bhuiyan


- [ ] **Objectives:**
- Facial detection
- gender classification
- emotion recognition
- landmark detection


##  B. ABSTRACT 

people express their feelings through their facial expression. like us we can creare a machine which can tell us about people's emotion by detecting their facial expression.Some of critical emotions are happy, sad, anger, disgust, fear, surprise etc. Facial expressions play a key role in non-verbal communication which appears due to internal feelings of a person that reflects on the faces.Nowadays, analysis of human body movements for emotion recognition is essential for social communication. Non-verbal communication methods like body movements, facial expression, gestures and eye movements are used in several applications.

 for many years,In order to computer modeling of human's emotion, a plenty of research has been accomplished. But still it is far behind from human vision system. In this paper, we are providing better approach to predict human emotions (Frames by Frames) using deep  and how emotion intensity changes on a face from low level to high level of emotion. The assessment through the proposed experiment confers quite good result and obtained accuracy may give encouragement to the researchers for future model of computer based emotion recognition system.


![Coding]![crm-emotion_recognition_mobile](https://user-images.githubusercontent.com/82527970/114949775-84b7d580-9e73-11eb-953f-9a4aa2e7128d.png)

Figure 1 shows the AI output of detecting facial emotions.


## C.  DATASET
 Facial Expressions plays an important role in interpersonal communication. Facial expression is a non verbal scientific gesture which gets expressed in our face as per our emotions.  And Automatic recognition of facial expression plays an important role in artificial intelligence and robotics .So, it is a need of the generation. Some application related to this include Personal identification and Access control,   Videophone   and      Teleconferencing,      Forensic   application,   Human-Computer Interaction,  Automated Surveillance, Cosmetology and so on. The objective of this project is to develop Automatic Facial Expression Recognition System which can take human facial images containing some expression as input and recognize and classify it into seven different expression class such as :

I.Neutral   II.Angry  III.Disgust  IV.FearV.Happy   VI.Sadness  VII.Surprise


![1-Figure2-2](https://user-images.githubusercontent.com/82527970/114954433-64d8df80-9e7c-11eb-83f6-59358eb26db8.png)

## MOTIVATION :
Significant  debate  has  risen  in  past  regarding  the  emotions  portrayed  in  the  world famous masterpiece of Mona Lisa. British Weekly „New Scientist‟ has stated that she is in fact a blend of many different emotions, 83%happy, 9% disgusted, 6% fearful, 2% angry.

![Mona_Lisa_(copy,_Hermitage)](https://user-images.githubusercontent.com/82527970/114955291-2d6b3280-9e7e-11eb-9ccc-265f0bd5489d.jpg)

We  have  also  been  motivated  observing  the  benefits  of  physically  handicapped people like deaf and dumb. But if any normal human being or an automated system can understand their needs by observing their facial expression then it becomes a lot easier  for  them  to  make  the  fellow  human  or  automated  system  understand  their needs.

![deaf__blind__dumb_by_biaayla-d5d2upj1](https://user-images.githubusercontent.com/82527970/114955624-e893cb80-9e7e-11eb-8f40-93646a43e43b.jpg)

## PROBLEM DEFINITION :
Human facial expressions can be easily classified into 7 basic emotions: happy, sad, surprise, fear, anger, disgust, and neutral. Our facial emotions are expressed through activation  of  specific  sets  of  facial  muscles.  These  sometimes  subtle,  yet  complex, signals in an expression often contain an abundant amount of information about our state of mind. Through facial emotion recognition, we are able to measure the effects that  content  and  services  have  on  the  audience/users  through  an  easy  and  low-cost procedure.   For   example,   retailers   may   use   these   metrics   to   evaluatecustomer interest.   Healthcare   providers   can   provide   better   service   by   using   additional information    aboutpatients'    emotional    stateduring    treatment.    Entertainment producers  can  monitoraudience  engagementin  events  to  consistently  create  desired content.Humans are well-trained in reading the emotions of others, in fact, at just 14 months old,   babies   can   already   tell   the   difference   between   happy   and   sad.But   can computers  do  a  better  job  than  us  in  accessing  emotional  states?To  answer  the question, Wedesigned a deep learning neural network that gives machines the ability to make inferences about our emotional states. In other words, wegive them eyes to see what we can see.
Problem formulation of our project:

![Screenshot (228)](https://user-images.githubusercontent.com/82527970/114956192-3230e600-9e80-11eb-89bd-0f2afb570636.png)

Facial  expression  recognition  is  a  process  performed  by  humans  or  computers, which consists of:
1.Locating  faces  in  the  scene  (e.g.,  in  an  image;  this  step  is  also  referred  to  as facedetection),
2.Extracting  facial  features  from  the  detected  face  region  (e.g.,  detecting  the  shape of facialcomponents or describing the texture of the skin in a facial area; this step is referred to asfacial feature extraction),
3.Analyzing  the  motion  of  facial  features  and/or  the  changes  in  the  appearance  of facialfeatures   and   classifying   this   information   into   some   facial-expression-interpretativecategoriessuch  as  facial  muscle  activations  like  smile  or  frown, emotion   (affect)categories   like   happiness   or   anger,   attitude   categories   like (dis)liking  or  ambivalence,  etc.(this  step  is  also  referred  to  as  facial  expression interpretation).Several Projects have already been done in this fields and our goal will not only be to develop a Automatic Facial Expression Recognition System but also improving the accuracy of this system compared to the other available systems


## D.   PROJECT STRUCTURE

The following directory is our structure of our project:
- $ tree --dirsfirst --filelimit 10
- .
- ├── dataset
- │   ├── [690 entries]
- │   └── [686 entries]
- ├── examples
- │   ├── example_01.png
- │   ├── example_02.png
- │   └── example_03.png
- ├── facial_emotion_detector
- │   ├── deploy.prototxt
- │   └── res10_300x300_ssd_iter_140000.caffemodel
- ├── detect_facial_emotion_image.py
- ├── detect_facial_emotion_video.py
- ├── facial_emotion_detector.model
- ├── plot.png
- └── train_facial_emotion_detector.py
- 5 directories, 10 files


The dataset/ directory contains the data described in the “Facial emotion detection dataset” section.



In the next two sections, we will train our facial emotion detector.



## E   TRAINING  FACIAL EMOTION DETECTION

We are now ready to train our facial emotion detector using Keras, TensorFlow, and Deep Learning.

From there, open up a terminal, and execute the following command:

- $ python train_facial_emotion_detector.py --dataset dataset
- [INFO] loading images...
- [INFO] compiling model...
- [INFO] training head...
- Train for 34 steps, validate on 276 samples
- Epoch 1/20
- 34/34 [==============================] - 30s 885ms/step - loss: 0.6431 - accuracy: 0.6676 - val_loss: 0.3696 - val_accuracy: 0.8242
- Epoch 2/20
- 34/34 [==============================] - 29s 853ms/step - loss: 0.3507 - accuracy: 0.8567 - val_loss: 0.1964 - val_accuracy: 0.9375
- Epoch 3/20
- 34/34 [==============================] - 27s 800ms/step - loss: 0.2792 - accuracy: 0.8820 - val_loss: 0.1383 - val_accuracy: 0.9531
- Epoch 4/20
- 34/34 [==============================] - 28s 814ms/step - loss: 0.2196 - accuracy: 0.9148 - val_loss: 0.1306 - val_accuracy: 0.9492
- Epoch 5/20
- 34/34 [==============================] - 27s 792ms/step - loss: 0.2006 - accuracy: 0.9213 - val_loss: 0.0863 - val_accuracy: 0.9688
- ...
- Epoch 16/20
- 34/34 [==============================] - 27s 801ms/step - loss: 0.0767 - accuracy: 0.9766 - val_loss: 0.0291 - val_accuracy: 0.9922
- Epoch 17/20
- 34/34 [==============================] - 27s 795ms/step - loss: 0.1042 - accuracy: 0.9616 - val_loss: 0.0243 - val_accuracy: 1.0000
- Epoch 18/20
- 34/34 [==============================] - 27s 796ms/step - loss: 0.0804 - accuracy: 0.9672 - val_loss: 0.0244 - val_accuracy: 0.9961
- Epoch 19/20
- 34/34 [==============================] - 27s 793ms/step - loss: 0.0836 - accuracy: 0.9710 - val_loss: 0.0440 - val_accuracy: 0.9883
- Epoch 20/20
- 34/34 [==============================] - 28s 838ms/step - loss: 0.0717 - accuracy: 0.9710 - val_loss: 0.0270 - val_accuracy: 0.9922

- [INFO] evaluating network...

![5-Table2-1](https://user-images.githubusercontent.com/82527970/114965571-aeccc000-9e92-11eb-9763-aba76eb16cf7.png)



![Figure 4]![images](https://user-images.githubusercontent.com/82527970/114965889-5518c580-9e93-11eb-90b0-4052063c694c.jpg)

Figure 4: Figure 10: Facial emotion detector training accuracy/loss curves demonstrate high accuracy and little signs of overfitting on the data

As you can see, we are obtaining ~99% accuracy on our test set.

Looking at Figure 4, we can see there are little signs of overfitting, with the validation loss lower than the training loss. 

Given these results, we are hopeful that our model will generalize well to images outside our training and testing set.


## F.  RESULT AND CONCLUSION

Detecting Facial emotion detection with OpenCV in real-time

You can then launch the facial emotion detector in real-time video streams using the following command:
- $ python detect_facial_emotion_video.py
- [INFO] loading face detector model...
- INFO] loading facial emotion detector model...
- [INFO] starting video stream...

[![Figure5](https://img.youtube.com/vi/uDRYr0CWSwI/0.jpg)](https://www.youtube.com/watch?v=uDRYr0CWSwI "Figure5")

Figure 5: Facial emotion detector in real-time video streams

In Figure 5, you can see that our Facial emotion detector is capable of running in real-time (and is correct in its predictions as well.



## G.   PROJECT PRESENTATION 

In this project, you learned how to create a facial emotion detector using OpenCV, Keras/TensorFlow, and Deep Learning.

We fine-tuned MobileNetV2 on our facial emotion dataset and obtained a classifier that is ~99% accurate.

We then took this facial emotion classifier and applied it to both images and real-time video streams by:

- Detecting emotions in images/video
- Extracting each individual emotion
- Applying our facial emotion classifier

Our facial emotion detector is accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient, making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, Jetosn, Nano, etc.).

[![demo](https://img.youtube.com/vi/AP9e4ny_KHc/0.jpg)](https://www.youtube.com/watch?v=AP9e4ny_KHc "demo")
