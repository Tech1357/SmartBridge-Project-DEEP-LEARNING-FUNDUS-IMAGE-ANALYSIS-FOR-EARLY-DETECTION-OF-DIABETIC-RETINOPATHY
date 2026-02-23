# Diabetic Retinopathy Detection System

A deep learning-based web application for detecting and classifying diabetic retinopathy severity from retinal fundus images using transfer learning with MobileNetV2 architecture.

## Project Overview

Diabetic Retinopathy is a diabetes complication that affects eyes and can lead to blindness if not detected early. This project implements an automated detection system that classifies retinal images into five severity levels, helping in early diagnosis and treatment planning.

## Features

- **User Authentication**: Secure registration and login system with IBM Cloudant database integration
- **Image Upload**: Support for PNG, JPG, and JPEG retinal fundus images
- **AI-Powered Classification**: Deep learning model trained to classify images into 5 severity levels
- **Real-time Predictions**: Instant classification results with confidence scores
- **Responsive UI**: Clean, user-friendly interface for seamless interaction
- **Prediction History**: Store and track prediction results in cloud database

## Classification Categories

The system classifies diabetic retinopathy into five severity levels:

1. **No_DR** - No Diabetic Retinopathy
2. **Mild** - Mild Non-Proliferative Diabetic Retinopathy
3. **Moderate** - Moderate Non-Proliferative Diabetic Retinopathy
4. **Severe** - Severe Non-Proliferative Diabetic Retinopathy
5. **Proliferate_DR** - Proliferative Diabetic Retinopathy

## Technology Stack

### Backend
- **Flask** - Python web framework
- **TensorFlow/Keras** - Deep learning framework
- **MobileNetV2** - Pre-trained CNN model for transfer learning
- **IBM Cloudant** - NoSQL cloud database for user data and predictions

### Frontend
- **HTML5/CSS3** - Structure and styling
- **Bootstrap** - Responsive design framework
- **JavaScript** - Client-side interactivity

### Machine Learning
- **Transfer Learning** - Using pre-trained MobileNetV2 on ImageNet
- **Data Augmentation** - Rotation, flipping, zooming for robust training
- **Fine-tuning** - Custom classification layers for diabetic retinopathy detection

## Project Structure

```
DL_Project/
│
├── app.py                          # Main Flask application
├── cloudant_config.py              # IBM Cloudant database configuration
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
│
├── model/
│   ├── Updated-Xception-diabetic-retinopathy.h5    # Trained model
│   ├── Xception_Diabetic_retinopathy.ipynb         # Training notebook
│   └── Test Transfer Learning Models.ipynb         # Model comparison notebook
│
├── templates/
│   ├── index.html                  # Home page
│   ├── login.html                  # Login page
│   ├── register.html               # Registration page
│   ├── prediction.html             # Prediction results page
│   └── logout.html                 # Logout confirmation page
│
├── static/
│   └── css/
│       └── style.css               # Custom styles
│
├── uploads/                        # Uploaded images storage
│
└── data/                          # Training dataset
    ├── Mild/
    ├── Moderate/
    ├── No_DR/
    ├── Proliferate_DR/
    └── Severe/
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- IBM Cloud account (optional, for Cloudant database)

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd DL_Project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure IBM Cloudant (Optional)**
   - Create an IBM Cloudant service on IBM Cloud
   - Update `cloudant_config.py` with your credentials:
   ```python
   CLOUDANT_USERNAME = "your-username"
   CLOUDANT_APIKEY = "your-api-key"
   DATABASE_NAME = "diabetic_retinopathy_db"
   ```

4. **Prepare the dataset**
   - Organize retinal images into folders by severity level
   - Place in the `data/` directory with subfolders: Mild, Moderate, No_DR, Proliferate_DR, Severe

5. **Train the model (if needed)**
```bash
python train_model.py
```

6. **Run the application**
```bash
python app.py
```

7. **Access the application**
   - Open browser and navigate to: `http://localhost:5000`

## Usage

### For Users

1. **Register/Login**
   - Create a new account or login with existing credentials
   - User data is securely stored in IBM Cloudant database

2. **Upload Image**
   - Navigate to the prediction page
   - Upload a retinal fundus image (PNG, JPG, or JPEG)
   - Maximum file size: 16MB

3. **View Results**
   - Get instant classification results
   - View confidence score for the prediction
   - See the uploaded image alongside results

4. **Predict Another**
   - Use "Predict Another Image" button to analyze more images

### For Developers

#### Model Training

The model uses transfer learning with MobileNetV2:

```python
# Key training parameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 5
```

#### Data Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    fill_mode='nearest'
)
```

#### Model Architecture

- Base: MobileNetV2 (pre-trained on ImageNet)
- Custom layers: GlobalAveragePooling2D, Dense layers with Dropout
- Output: 5-class softmax classification
- Optimizer: Adam
- Loss: Categorical Crossentropy

## Model Performance

The model is trained on a comprehensive dataset of retinal fundus images with the following distribution:

- **No_DR**: 1,805 images
- **Moderate**: 999 images
- **Mild**: 370 images
- **Proliferate_DR**: 295 images
- **Severe**: 193 images

**Total Dataset**: 3,662 images

The model achieves reliable performance across all five severity levels, making it suitable for clinical decision support and screening applications.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` or `/index` | GET | Home page |
| `/register` | GET, POST | User registration |
| `/login` | GET, POST | User login |
| `/logout` | GET | User logout |
| `/predict` | GET, POST | Image upload and prediction |

## Database Schema

### Users Collection
```json
{
  "_id": "user@email.com",
  "name": "User Name",
  "password": "hashed_password",
  "registered_date": "2026-02-21T10:30:00"
}
```

### Predictions Collection
```json
{
  "user": "user@email.com",
  "prediction": "Moderate",
  "confidence": "85.50",
  "image_name": "20260221_103000_image.png",
  "timestamp": "2026-02-21T10:30:00",
  "all_probabilities": {
    "Mild": "5.20",
    "Moderate": "85.50",
    "No_DR": "3.10",
    "Proliferate_DR": "4.00",
    "Severe": "2.20"
  }
}
```

## Security Features

- Session-based authentication
- Secure file upload with extension validation
- File size limits to prevent abuse
- Secure filename handling with werkzeug
- Password storage (recommend adding hashing in production)

## Future Enhancements

- [ ] Password hashing with bcrypt
- [ ] Email verification for registration
- [ ] Export prediction reports as PDF
- [ ] Batch image processing
- [ ] Model performance metrics dashboard
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Integration with hospital management systems

## Requirements

```
flask==2.3.0
tensorflow==2.15.0
numpy==1.24.3
pillow==10.0.0
werkzeug==2.3.0
cloudant==2.15.0
scikit-learn==1.3.0
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is developed for educational and research purposes.

## Acknowledgments

- Dataset: Diabetic Retinopathy Detection Challenge
- Pre-trained model: MobileNetV2 from TensorFlow/Keras
- Cloud database: IBM Cloudant
- Framework: Flask web framework

## Contact

For questions, issues, or suggestions, please open an issue in the repository.

---

**Note**: This system is designed for research and educational purposes. For clinical use, please consult with medical professionals and ensure compliance with healthcare regulations.
