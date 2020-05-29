# ML-Devops

CONTAINERIZATION-Launching OS in a single click!

Creating docker images using the Dockerfile for any of the models to train with all the required libraries installed in them.

I have created two images one for the CNN and another one for the usual ML problems.


Creating a Job Chain of Job1, Job2, Job3, Job4 and Job5 using build pipeline plugin in Jenkins 

JOB 1: As soon as the developer pushes the code on GitHub, it should be automatically downloaded in the Jenkins workspace :

Linking with the GitHub Repo and copying it to our folder in RHEL8.

Launching the OS using Containerization Technology

JOB 2: By looking at the code or program file, Jenkins should automatically start the respective machine learning software installed interpreter and install the image container as well to deploy the code and start training.

( For example: If code uses CNN, then Jenkins should start the container that has already installed all the software required for the CNN processing).

JOB 3: Train the model and predict the accuracy.

JOB 4: If the accuracy of the model trained is less than our requirement ( here accuracy>=80%). 
It should be retrained, with the hyperparameters being changed.
Initially,
no_of_filters=32
kernel_size=3
pool_size=2
i=1

If the accuracy of the model is not as per our expectations, these values would be increased to get better learning and accuracy.

If accuracy is reached (90% in this case), the developer would be notified about the accuracy  of the model via email.
JOB 5: Create One extra to monitor i.e. 
If the container where the app is running fails due to any of the reasons. 
Then this job should automatically start the container again from where the last trained model left.


