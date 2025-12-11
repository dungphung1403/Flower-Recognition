📁 Project Structure
├── app.py                  # Streamlit web app to upload & classify images
├── flower_recognition.py   # Model training script
├── train/                  # Dataset (5 flower categories)
├── flower_names.bin        # Serialized trained model (generated after training)
├── tempDir/                # Temp folder used for uploaded images

🧠 1. Model Training – flower_recognition.py

This script handles:

✔ Cleaning the dataset

It removes corrupted JPEG images using a JFIF signature check.

✔ Loading and preprocessing the dataset

Using image_dataset_from_directory

80% training / 20% validation

180×180 px resizing

Prefetching + caching for performance

✔ Data Augmentation

Random flip, rotation, and zoom to improve generalization.

✔ Building the CNN model

A simple but effective Convolutional Neural Network:

Conv2D → MaxPool

Conv2D → MaxPool

Conv2D → MaxPool

Dropout

Dense classifier (5 output classes)

✔ Training
epochs = 15
optimizer = Adam
loss = SparseCategoricalCrossentropy

✔ Saving the model

The trained Keras model is serialized using pickle:

with open('flower_names.bin', 'wb') as f_out:
    pickle.dump(model, f_out)


Output file is used later by the Streamlit app.

💻 2. Prediction Web App – app.py

This is a Streamlit application for uploading an image and predicting its flower type.

✔ Features

Upload any flower image (.jpg, .png…)

Shows the uploaded image

Runs model prediction

Displays:

Flower type

Confidence score

✔ How it works

Loads the trained model from flower_names.bin

Resizes input images to 180×180

Computes softmax probabilities

Maps the predicted index to:

['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

✔ Running the web app
streamlit run app.py

📦 Installation
1. Clone the repository
git clone <your-repo-url>
cd <repo>

2. Install dependencies
pip install -r requirements.txt


Typical requirements include:

streamlit
tensorflow
keras
numpy
pickle-mixin
matplotlib

🚀 Training the Model

To retrain the model:

Place dataset inside the train/ folder with subfolders:

train/
  daisy/
  dandelion/
  rose/
  sunflower/
  tulip/


Run:

python flower_recognition.py


A new flower_names.bin file will be generated.

🌐 Running the Streamlit App

After training:

streamlit run app.py


Then open the URL displayed in the terminal (http://localhost:5000).

📌 Notes

tempDir/ must exist or be created automatically for file uploads.

If you want to switch to model.h5 instead of pickle, update both scripts accordingly.

Serialized Keras models via pickle work but .h5 or SavedModel is safer for production.
