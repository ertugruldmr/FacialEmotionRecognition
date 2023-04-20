<div style="position: absolute; top: 0; right: 0;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

# Face Emotion Recognition
 
## __Table Of Content__
- (A) [__Brief__](#brief)
  - [__Project__](#project)
  - [__Data__](#data)
  - [__Demo__](#demo) -> [Live Demo](https://ertugruldemir-facialemotionrecognition.hf.space)
  - [__Study__](#problemgoal-and-solving-approach) -> [Colab](https://colab.research.google.com/drive/14DrNsiYIRYUr3PKhtgF0wB5htVh15wK1)
  - [__Results__](#results)
- (B) [__Detailed__](#Details)
  - [__Abstract__](#abstract)
  - [__Explanation of the study__](#explanation-of-the-study)
    - [__(A) Dependencies__](#a-dependencies)
    - [__(B) Dataset__](#b-dataset)
    - [__(C) Modelling__](#e-modelling)
    - [__(D) Deployment as web demo app__](#g-deployment-as-web-demo-app)
  - [__Licance__](#license)
  - [__Connection Links__](#connection-links)
  - __NOTE__: 
  	- The model file exceeded limitations. you can download it from this [link](https://huggingface.co/spaces/ErtugrulDemir/FacialEmotionRecognition/resolve/main/Basic_EmotionModel.zip). The other demo links are also included the model download link.

## __Brief__ 

### __Project__ 
- This is an __image classification__ project that uses the  [__Facial-Expression-Datast__](https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset) to __classify the Face images__ into corresponding emotion specie.
- The __goal__ is build a deep learning image classification model that accurately __classify the Face images__ into corresponding emotion class.
- The performance of the model is evaluated using several __metrics__ loss and accuracy metrics.

#### __Overview__
- This project involves building a deep learning model to classfy the images. The dataset contains 35887 images with 7 classes. 28709 train and 7178 test images. The models selected according to model tuning results, the progress optimized respectively the previous tune results. The project uses Python and several popular libraries such as Pandas, NumPy, tensorflow.

#### __Demo__

<div align="left">
  <table>
    <tr>
    <td>
        <a target="_blank" href="https://ertugruldemir-facialemotionrecognition.hf.space" height="30"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30">[Demo app] HF Space</a>
      </td>
      <td>
        <a target="_blank" href="https://colab.research.google.com/drive/14DrNsiYIRYUr3PKhtgF0wB5htVh15wK1"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Demo app] Run in Colab</a>
      </td>
      <td>
        <a target="_blank" href="https://github.com/ertugruldmr/FacialEmotionRecognition/blob/main/study.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">[Traning pipeline] source on GitHub</a>
      </td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/14DrNsiYIRYUr3PKhtgF0wB5htVh15wK1"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Traning pipeline] Run in Colab</a>
      </td>
    </tr>
  </table>
</div>


- Description
    - __classify the Face images__ into one of the 7 emotion class correctly.
    - __Usage__: Set the feature values through sliding the radio buttons then use the button to predict.
- Embedded [Demo](https://ertugruldemir-facialemotionrecognition.hf.space) window from HuggingFace Space
    

<iframe
	src="https://ertugruldemir-facialemotionrecognition.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

#### __Data__
- The [__Facial-Expression-Datast__](https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset) from kaggle dataset api.
- The dataset contains 35887 images with 7 classes. 28709 train and 7178 test images. 
- The dataset contains the following features:
  - Example Dataset
      <div style="text-align: center;">
        <img src="docs/images/example_images.png" style="width: 400px; height: 300px;">
      </div>
  - Data Augmentation
      <div style="text-align: center;">
        <img src="docs/images/agumentation.png" style="width: 400px; height: 300px;">
      </div>


#### Problem, Goal and Solving approach
- This is a __Image classification__ problem  that uses the  [__Facial-Expression-Datast__](https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset)  to __classify the Face images_.
- The __goal__ is to build a model that accurately __classify the Face images__ into corresponding emotion class.
- __Solving approach__ is that using the supervised deep learning models. A basic custom conv model is used but it was not enough so after several transfer learning implementation, 'MobileNetV2' state of art model tuned. 

#### Study
The project aimed predict the house prices using the features. The study includes following chapters.
- __(A) Dependencies__: Installations and imports of the libraries.
- __(B) Dataset__: Downloading and loading the dataset. Preparing the dataset via tensorflow dataset api. Configurating the dataset performance and related pre-processes. 
- __(C) Preprocessing__: Type casting, value range scaling, resizing, Implementing augmentation methods on train dataset and image classification related processes.
- __(D) Modelling__:
  - Basic Modelling
    - Basic Custom Convolutional Deep learning model as classifier.
    - I also implemented tranfer learning approach to compare.
  - Transfer Learning
    - Selected base model has been tuned via adding fully connected classifier layer.
    - These base models are MobileNetV2, VGG16, ResNet50,EfficientNetB0. In this problem, MobileNetV2 is the better performance and results compared others.
  - Predicting
    - Implementing the model on the example data, inferencing.
  - Evaluating
    - Saving the model architecture with weights.
- __(D) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

#### results
- The final model is __Custom Classifier Network__ because of the results and less complexity.
  - Custom Conv Neural Network Classfier Results
        <table><tr><th>Classification Results </th><th></th></tr><tr><td>
      | model                         | loss   | accuracy |
      |-------------------------------|--------|----------|
      | [val] Basic Cusom Classifier  | 1.4308 | 0.4683   |
      | [val] Fine-Tuned MobileNetV2  | 1.8712 | 0.3797   |
      </td></tr></table>

## Details

### Abstract
- [__Facial-Expression-Datast__](https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset) is used to classify the Face images into corresponding emotion class. The dataset contains 35887 images with 7 classes. 28709 train and 7178 test images. The problem is a supervised learning task as classification with multiple class. The goal is Face images into corresponding emotion  class using  through supervised custom deep learning algorithms or related training approachs of pretrained state of art models .The study includes creating the environment, getting the data, preprocessing the data, exploring the data, agumenting the data, modelling the data, saving the results, deployment as demo app. Training phase of the models implemented through tensorflow callbacks. After the custom model traininigs, transfer learning and fine tuning approaches are implemented. Selected the basic and more succesful when comparet between other models  is  basic custom deep learning classifier.__Custom Classifier__ model has __1.4308__ loss , __0.4683__ acc,  other metrics are also found the results section. Created a demo at the demo app section and served on huggingface space.  


### File Structures

- File Structure Tree
```bash
├── demo_app
│   ├── app.py
│   ├── Basic_EmotionModel
│   │   ├── assets
│   │   ├── fingerprint.pb
│   │   ├── keras_metadata.pb
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── requirements.txt
│   └── sample_images
├── docs
│   └── images
├── env
│   ├── env_installation.md
│   └── requirements.txt
├── readme.md
└── study.ipynb
```
- Description of the files
  - demo_app/
    - Includes the demo web app files, it has the all the requirements in the folder so it can serve on anywhere.
  - demo_app/Basic_EmotionModel:
    - Custom Convolutional Classifier model.
  - demo_app/sample_images
    - Example cases to test the model.
  - demo_app/requirements.txt
    - It includes the dependencies of the demo_app.
  - docs/
    - Includes the documents about results and presentations
  - env/
    - It includes the training environmet related files. these are required when you run the study.ipynb file.
  - LICENSE.txt
    - It is the pure apache 2.0 licence. It isn't edited.
  - readme.md
    - It includes all the explanations about the project
  - study.ipynb
    - It is all the studies about solving the problem which reason of the dataset existance.    

### Explanation of the Study
#### __(A) Dependencies__:
  - There is an additional installation which is kaggle dataset api, the other dependencies are included the libraries. The libraries which already installed on the environment are enough. You can create an environment via env/requirements.txt. Create a virtual environment then use hte following code. It is enough to satisfy the requirements for runing the study.ipynb which training pipeline.
  - Dataset can download from tensoflow.
#### __(B) Dataset__: 
  - Downloading the [__Facial-Expression-Datast__](https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset)  via tensorflow dataset api. 
  - The dataset contains 35887 images with 7 classes. 28709 train and 7178 test images.
  - Preparing the dataset via resizing, scaling into 0-1 value range, implementing data augmentation and etc image preprocessing processes. 
  - Creating the tensorflow dataset object then configurating.
  - Example Dataset
      <div style="text-align: center;">
        <img src="docs/images/example_images.png" style="width: 400px; height: 300px;">
      </div>
  - Data Augmentation
      <div style="text-align: center;">
        <img src="docs/images/agumentation.png" style="width: 800px; height: 600px;">
      </div>
  - Class Distributions
      <div style="text-align: center;">
        <img src="docs/images/target_dists.png" style="width: 400px; height: 600px;">
      </div>
      <div style="text-align: center;">
        <img src="docs/images/target_dist_pie.png" style="width: 400px; height: 150px;">
      </div>

#### __(C) Modelling__: 
  - The processes are below:
    - Archirecture
      - Basic Cusom Classifier
        <div style="text-align: center;">
          <img src="docs/images/CustomBasicClassifierModel.png" style="width: 400px; height: 900px;">
        </div>
      - Fine-Tuned MobileNetV2
        - Classifier Architecture
          <div style="text-align: center;">
            <img src="docs/images/tranfer_learning_architecture.png" style="width: 400px; height: 300px;">
          </div>
        - Base Model (ResNetV2) of the classifier Architecture
          <div style="text-align: center;">
            <img src="docs/images/transfer_learning_base_model_arhitecture.png" style="width: 400px; height: 8000px;">
          </div>
    - Training
      - Basic Cusom Classifier
        <div style="text-align: center;">
          <img src="docs/images/training_hist.png" style="width: 800px; height: 600px;">
        </div>
      - Fine-Tuned MobileNetV2
        <div style="text-align: center;">
          <img src="docs/images/transfer_learning_history.png" style="width: 800px; height: 600px;">
        </div>
    - Evaluating and classification results
      - Custom Conv Neural Network Classfier Results
        <table><tr><th>Classification Results </th><th></th></tr><tr><td>
      | model                         | loss   | accuracy |
      |-------------------------------|--------|----------|
      | [val] Basic Cusom Classifier  | 1.4308 | 0.4683   |
      | [val] Fine-Tuned MobileNetV2  | 1.8712 | 0.3797   |
      </td></tr></table>
  - Saving the project and demo studies.
    - trained model __Basic_EmotionModel__ as tensorflow (keras) saved_model format.

#### __(D) Deployment as web demo app__: 
  - Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.
  - Desciption
    - Project goal is classifiying the images into corresponding class.
    - Usage: upload or select the image for classfying then use the button to predict.
  - Demo
    - The demo app in the demo_app folder as an individual project. All the requirements and dependencies are in there. You can run it anywhere if you install the requirements.txt.
    - You can find the live demo as huggingface space in this [demo link](https://ertugruldemir-facialemotionrecognition.hf.space) as full web page or you can also us the [embedded demo widget](#demo)  in this document.  
    
## License
- This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

<h1 style="text-align: center;">Connection Links</h1>

<div style="text-align: center;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://www.hackerrank.com/ertugrulbusiness"><img src="https://hrcdn.net/fcore/assets/work/header/hackerrank_logo-21e2867566.svg" height="30"></a>
    <a href="https://app.patika.dev/ertugruldmr"><img src="https://app.patika.dev/staticFiles/newPatikaLogo.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

