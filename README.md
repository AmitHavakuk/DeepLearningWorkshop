ReadMe
Demo:


![Alt text](/preview/verification_prev1.png) ![Alt text](/preview/verification_prev2.png)

![Alt text](/preview/classification_prev1.png) ![Alt text](/preview/classification_prev2.png)




To run our code, you need to pip install the following modules:

We used python 3.10 64x and the following versions (which we recommend to use):
customtkinter==5.2.2
dlib==19.24.6
matplotlib==3.9.2
numpy==1.26.3
opencv_python==4.10.0.84
Pillow==10.2.0
pytorch_metric_learning==2.7.0
scikit_learn==1.5.2
seaborn==0.13.2
torch==2.5.1+cu124
torchvision==0.20.1+cu124


You need to install torch and torchvision with cuda version that suits your GPU (we recommend using cuda and not cpu), see https://pytorch.org/
If you have problems installing dlib try downloading visual studio and you insure you 64x python. Note that if you use NumPy with version >= 2.0.0 you need to change np.Inf to np.inf. 


Structure:


verification_GUI - instructions

Our application allows for side-by-side image verification.
The GUI is rather simple. The image on the left shows how the GUI appears when the user opens it.
The user first chooses the model he would like to use from the options menu. The default is the sunglasses model we described, but he can also choose one of the extensions which we describe later.
image 1 - upload locally from your computer or use the webcam button for a pop-up window that allows for snapping a photo from live feed. We make sure to align and crop the picture around your given face for optimal performance.
image 2 - same instructions as image 1.
After having loaded two pictures, results will be displayed as seen in the image on the right.
resulting model answer: Same Person = True or False, depending on whether similarity > threshold or not.
similarity (distance) between the embeddings our model gave for the pictures. Can be portrayed as a confidence level.
threshold We chose to use a higher threshold than the one that produces the best results on our test histograms. This decision was based on testing images captured from a webcam, which are of lower quality and may be darker or blurrier. In these cases, we observed that a higher threshold performs better. Furthermore, since verification prioritizes minimizing the false positive rate (FPR) over the false negative rate (FNR), a higher threshold aligns with this goal. Additionally, it is worth noting that the images in our dataset are of celebrities, where facial lighting is flattering, and the faces are clear. Therefore, we expect less optimal results on webcam images. To perform a good simulation of the model, we recommend running it on images from our test dataset.


classification_GUI - instructions

We added support for image classification. Given N constant pictures of different identities, we would now like to classify a new picture for one of them. We do this by calculating the embedding of the new picture and finding which of the N pictures has the closest embedding to it. If the distance to the first closest and second closest matches is smaller than some threshold, we decide there is no match, otherwise take the first match. GUI instructions: upload one image of each identity to the classification_images folder, run the program and press calculate embeddings (this will take some time), then upload/capture a new picture and watch the results.


Code for spliting the data to identity folders
Code for adding sunglasses, augmentations and masks
Training code – This code uses Epoch_17.pt which is a file containing the initial weights for MobileFaceNets. This file was too big for us to upload to github.

