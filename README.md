# 🌿 Plant Disease Recognition System

A deep learning-based web application that detects diseases in crop leaves from uploaded images. Built using two CNN architectures — a custom CNN trained from scratch and a fine-tuned EfficientNetB4 using transfer learning — deployed as an interactive Streamlit web app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://plant-disease-detection-wahrbqbm6uhj3wmgrgrqty.streamlit.app/)

**🔗 Live Demo:** https://plant-disease-detection-wahrbqbm6uhj3wmgrgrqty.streamlit.app/

---

## 📸 Screenshots

> Disease Recognition page — upload a leaf image and get instant prediction

![App Screenshot](home_page.jpeg)

---

## 🚀 Features

- Upload any crop leaf image and get instant disease classification
- Detects 38 disease categories across 14 crop types
- Supports healthy vs diseased classification
- Fast inference — results in seconds
- Clean 3-page interface: Home, About, and Disease Recognition

---

## 🧠 Models

### Model 1 — Custom CNN (Trained from Scratch)
- 5 convolutional blocks with increasing filter depth: 32 → 64 → 128 → 256 → 512
- MaxPooling after each block for spatial downsampling
- Dropout layers (0.25 and 0.4) to prevent overfitting
- Dense layer with 1500 units + Softmax output for 38 classes
- Input size: 128×128 RGB images
- Optimizer: Adam (lr=0.0001), Loss: Categorical Crossentropy
- Trained for 10 epochs on Kaggle with Tesla P100 GPU

### Model 2 — EfficientNetB4 (Transfer Learning)
- Pretrained EfficientNetB4 base model loaded with ImageNet weights
- Custom head: GlobalAveragePooling2D → Dropout → Dense(512) → Dense(38, softmax)
- Total parameters: 18.6M (18.5M trainable)
- Input size: 380×380 RGB images
- Fine-tuned end-to-end with Adam (lr=0.0001)
- **Frozen base validation accuracy: 98%+** | **Fine-tuned validation accuracy: 99%+** | **Overall Classification Accuracy: 95%**
- Significant accuracy improvement over training from scratch with far fewer epochs required

---

## 📊 Results

| Model | Validation Accuracy | Notes |
|-------|-------------------|-------|
| Custom CNN | **~98%** | Trained from scratch, 10 epochs |
| EfficientNetB4 (frozen base) | **98%+** | Transfer learning, base layers frozen |
| EfficientNetB4 (fine-tuned) | **99%+** | Full model fine-tuned end-to-end |

### Classification Report Highlights (EfficientNetB4)
- Overall accuracy: **95%** across 17,572 validation images
- Macro avg precision: **0.96** | Recall: **0.95** | F1: **0.95**
- Best performing classes: Grape (healthy, Esca), Cherry (healthy), Corn (healthy) — F1 > 0.99
- 38 disease categories classified with high confidence

---

## 🌱 Supported Crops and Diseases

| Crop | Diseases Detected |
|------|------------------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Peach | Bacterial Spot, Healthy |
| Cherry | Powdery Mildew, Healthy |
| Strawberry | Leaf Scorch, Healthy |
| + more | Blueberry, Raspberry, Soybean, Squash, Orange |

---

## 📁 Dataset

**New Plant Diseases Dataset** (Augmented)
- ~87,000 RGB images of healthy and diseased crop leaves
- 38 classes across 14 crop types
- 80/20 train-validation split
- Created using offline augmentation from the original PlantVillage dataset

| Split | Images |
|-------|--------|
| Train | 70,295 |
| Validation | 17,572 |
| Test | 33 |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Deep Learning | TensorFlow 2.20, Keras |
| Transfer Learning | EfficientNetB4 (ImageNet weights) |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn (classification report, confusion matrix) |
| Web App | Streamlit |
| Training Environment | Kaggle (Tesla P100 GPU) |

---

## 🏃 Run Locally

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/des-adi/plant-disease-detection.git
cd plant-disease-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

### Requirements
```
tensorflow==2.20.0
numpy
streamlit
Pillow
```

---

## 📂 Project Structure

```
plant-disease-detection/
├── main.py                          # Streamlit application
├── trained_model.keras              # Trained EfficientNetB4 model
├── home_page.jpeg                   # Home page banner image
├── requirements.txt                 # Python dependencies
├── plant-disease-detection.ipynb    # Training notebook (EfficientNetB4)
└── README.md
```

---

## 💡 How It Works

1. **Upload** a leaf image on the Disease Recognition page
2. Image is **preprocessed** — resized to 128×128, converted to array, normalized
3. **Model inference** — loaded EfficientNetB4 predicts probabilities across 38 classes
4. **Result** — class with highest probability is returned as the disease prediction

```python
def model_prediction(test_img):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_img, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index
```

---

## 🔍 Key Learnings

- **Transfer learning advantage**: EfficientNetB4 with frozen base achieved 98%+ validation accuracy, and 99%+ after full fine-tuning — compared to custom CNN which reached ~98% after 10 full epochs of training from scratch, demonstrating why pretrained models are preferred for image classification tasks with limited compute
- **EfficientNet architecture**: Uses compound scaling (depth + width + resolution) making it more parameter-efficient than VGG or ResNet equivalents
- **Deployment pipeline**: Learned to resolve Python version compatibility issues (TF 2.20 requires Python 3.12+) when deploying to Streamlit Cloud

---

## 👤 Author

**Aditya Deshmukh**
- GitHub: [@des-adi](https://github.com/des-adi)
- LinkedIn: [Aditya Deshmukh](https://linkedin.com/in/aditya-deshmukh)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
