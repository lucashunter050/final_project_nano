# Project Name

Many people wonder what type of plane they are boarding while waiting in the airport. My project is able to classify 4 of the most popular airliners in service today, the Airbus A320 and A350 and the Boeing 737 and 787. Our classification accuracy converged to 58% on the test data used; although, we only ran the model for one hour. Users can take a photo of the plane they are about to board and upload it to their Jetson Nano to receive rapid classifcation. Enthusiasts can also use this project to identify unknown airliners they see in the sky.

![p8](https://github.com/user-attachments/assets/24c4c46b-70cc-4603-8aa6-4b12505c2275)


the image below shows the result of the classifier on a Condor A320. The model correctly assigns the aircraft to the a320 label with 53% confidence.


![Screenshot 2024-07-25 at 9 48 34 AM](https://github.com/user-attachments/assets/b765307a-6267-48ea-bf21-35d63da3ed16)


![Screenshot 2024-07-25 at 9 49 34 AM](https://github.com/user-attachments/assets/bbf13ade-8ee7-49c7-9997-176471d5d08a)


## The Algorithm

Add an explanation of the algorithm and how it works. Make sure to include details about how the code works, what it depends on, and any other relevant info. Add images or other descriptions for your project here. 

Our algorithm works as a classification neural network. We used transfer learning to retrain the resnet-18 based imagenet classifier. We ran training on over 300 images of each type of aircraft. Over the course of one hour, our test accuracy reached 58% over 34 epochs. Our model started at 15% accuracy. If we were able to run our training for longer, we likely would have reached much higher accuracy. 

## Running this project

1. Navigate to the jetson-inference/python/training/classification
2. set bash environment variables `NET=/models/planes2` and `DATASET=/data/planes`
3. run the command `imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/NAME_OF_CLASS_FOLDER/NAME_OF_JPEG DESIRED_NAME_OF_OUTPUT_JPG`
4. the classified image will appear in the DESIRED_NAME_OF_OUTPUT_JPG provided in the previous step. It will also be output to the terminal as a result of the previous command.

