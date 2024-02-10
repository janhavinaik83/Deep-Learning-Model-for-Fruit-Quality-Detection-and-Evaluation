# Deep-Learning-Model-for-Fruit-Quality-Detection-and-Evaluation


DESCRIPTION
India cultivates a diverse range of fruits and it is important to check its quality as it plays a critical role in the health of consumers. The utilization of pre-existing models such as ResNet50 for classification purposes showcases ramarkable efficiency and proficiency, yielding highly accurate results. Leveraging the expertise and learned features of these pre-trained models substancially enhances the system's performance.

RESNET50 MODEL 
The ResNet50 model is a popular deep neural network architecture commomly used for various computer vision tasks, including image classification. ResNet stands as a profound neural network architecture that revolutionized the notion of residual learning. It incorporates skip connections or shortcuts, enabling the network to leap over certain layers, thus facilitating the training of exceptionally deep networks. By introducing these skip connections, ResNet effectively addresses the vanishing gradient issue, empowering the training of deeper models that exhibit enhanced performance.

METHODOLOGY 
The proposed system is designed to overcome problems of manual techniques. This system consists of several steps to detect the quality of fruit using CNN architectural methods using pre- trained models. There are seven steps in the proposed model for quality detection as shown below.

Segregation of Dataset
The Dataset is segregated based on the quality of fruits. It contains three distinct categories namely Good Quality, Bad Quality and Mixed Quality. The total number of images is 12069 for training the model.

Splitting the dataset
We have divided our dataset into a training set,testing set and a validation set. In our case, out of the total 12,069 files, we have allocated 9,656 files (approximately 80% of the dataset) for training the model. This larger portion of data will be used to train the model and adjust its parameters to minimize the training loss and improve its ability to classify fruit quality accurately.

The remaining 2,413 files (approximately 20% of the dataset) have been reserved for testing. These files will be used to assess the model's performance on unseen data, allowing you to measure its accuracy, precision, recall, or any other evaluation metrics relevant to your task.

By splitting the dataset into a training set and a testing set, we can train and assess the performance of the ResNet-50 model on different subsets of data, helping us gain confidence in its ability to classify fruit quality with precision.

Load the ResNet Model
Train the Model
Train the Model Evaluation
