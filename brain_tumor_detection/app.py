from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img #preprocess uploaded images
import os 
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # uploading image


# model loading
model = load_model('brain_tumor_model.h5')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))  # Resize to 150x150
    img = img_to_array(img)  # Convert to array
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 150, 150, 3)
    img = img / 255.0  # Normalize
    return img


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Preprocess and predict
            img = preprocess_image(file_path)
            prediction = model.predict(img)
            result = "Tumor Detected" if prediction[0][0] >= 0.5 else "No Disease Detected"
            
            return render_template('index.html', prediction=result, image=filename)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])  #Ensure the upload folder exists
    app.run(debug=True)
