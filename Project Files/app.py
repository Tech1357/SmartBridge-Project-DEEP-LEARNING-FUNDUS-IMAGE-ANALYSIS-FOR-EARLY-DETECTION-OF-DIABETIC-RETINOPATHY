from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np

# Import Cloudant
try:
    from cloudant.client import Cloudant
    from cloudant.error import CloudantException
    from cloudant.result import Result, ResultByKey
    import cloudant_config
    CLOUDANT_AVAILABLE = True
except ImportError:
    CLOUDANT_AVAILABLE = False
    print("⚠️  Cloudant library not installed. Install with: pip install cloudant")

app = Flask(__name__)
app.secret_key = 'diabetic_retinopathy_secret_key_2026'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'model/Updated-Xception-diabetic-retinopathy.h5'
IMG_SIZE = 224  # Updated to match MobileNetV2 input size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model will be loaded after training
model = None

# Cloudant Database
my_database = None

# Class labels
CLASS_LABELS = {
    0: 'Mild',
    1: 'Moderate',
    2: 'No_DR',
    3: 'Proliferate_DR',
    4: 'Severe'
}

# Initialize Cloudant Connection
def init_cloudant():
    global my_database
    if not CLOUDANT_AVAILABLE:
        print("⚠️  Cloudant not available")
        return False
    
    try:
        # IBM Cloud Identity & Access Management
        client = Cloudant.iam(
            cloudant_config.CLOUDANT_USERNAME,
            cloudant_config.CLOUDANT_APIKEY,
            connect=True
        )
        
        # Create database if not exists
        if cloudant_config.DATABASE_NAME not in client.all_dbs():
            my_database = client.create_database(cloudant_config.DATABASE_NAME)
            print(f"✅ Database '{cloudant_config.DATABASE_NAME}' created")
        else:
            my_database = client[cloudant_config.DATABASE_NAME]
            print(f"✅ Connected to database '{cloudant_config.DATABASE_NAME}'")
        
        return True
    except Exception as e:
        print(f"⚠️  Cloudant connection failed: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_if_exists():
    global model
    if os.path.exists(MODEL_PATH) and model is None:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model as keras_load_model
            print("Loading model...")
            model = keras_load_model(MODEL_PATH)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return model is not None

def preprocess_image(img_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
@app.route('/index')
def index():
    # Check if user is logged in
    if 'user' in session:
        # User is logged in
        return render_template('index.html', 
                             pred='Logout', 
                             vis='hidden', 
                             vis2='visible',
                             msg=f"Welcome, {session.get('name', 'User')}!")
    else:
        # User is not logged in
        return render_template('index.html', 
                             pred='Login', 
                             vis='visible', 
                             vis2='hidden',
                             msg='')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate passwords match
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html', error='Passwords do not match')
        
        # Store data in dictionary
        data = {
            '_id': email,
            'name': name,
            'password': password,
            'registered_date': datetime.now().isoformat()
        }
        
        # Check if Cloudant is available
        if my_database is not None:
            try:
                # Check if user already exists
                query = {'_id': {'$eq': email}}
                docs = my_database.get_query_result(query)
                
                if len(docs.all()) == 0:
                    # User doesn't exist, register new user
                    my_database.create_document(data)
                    flash('Registration successful! Please login.', 'success')
                    return redirect(url_for('login'))
                else:
                    # User already exists
                    flash('User already registered. Please login.', 'warning')
                    return render_template('register.html', 
                                         error='User already registered. Please login.')
            except Exception as e:
                print(f"Database error: {e}")
                flash('Registration failed. Please try again.', 'error')
                return render_template('register.html', error='Registration failed')
        else:
            # Cloudant not available, allow registration anyway
            session['user'] = email
            session['name'] = name
            flash('Registration successful! (Database not connected)', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if Cloudant is available
        if my_database is not None:
            try:
                # Validate credentials
                query = {'_id': {'$eq': email}}
                docs = my_database.get_query_result(query)
                docs_list = docs.all()
                
                if len(docs_list) == 0:
                    # Username not found
                    flash('Username not found. Please register.', 'error')
                    return render_template('login.html', 
                                         error='Username not found. Please register.')
                else:
                    # Check password
                    user_data = docs_list[0]
                    if user_data['password'] == password:
                        # Login successful
                        session['user'] = email
                        session['name'] = user_data.get('name', 'User')
                        flash('Login successful!', 'success')
                        return redirect(url_for('index'))
                    else:
                        # Incorrect password
                        flash('Incorrect password. Please try again.', 'error')
                        return render_template('login.html', 
                                             error='Incorrect password')
            except Exception as e:
                print(f"Database error: {e}")
                flash('Login failed. Please try again.', 'error')
                return render_template('login.html', error='Login failed')
        else:
            # Cloudant not available, allow login anyway
            session['user'] = email
            session['name'] = 'User'
            flash('Login successful! (Database not connected)', 'success')
            return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return render_template('logout.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Check if user is logged in
    if 'user' not in session:
        flash('Please login to use prediction.', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        # Show prediction page with upload form
        return render_template('prediction.html', 
                             pred='Logout', 
                             vis='hidden', 
                             vis2='visible',
                             prediction=None)
    
    if not load_model_if_exists():
        flash('Model not found. Please train the model first.', 'error')
        return render_template('prediction.html', error='Model not found. Please train the model first.')
    
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return render_template('prediction.html', error='No file uploaded')
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return render_template('prediction.html', error='No file selected')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess and predict
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class]) * 100
        
        prediction_result = CLASS_LABELS[predicted_class]
        
        # Store prediction in Cloudant database
        if my_database is not None:
            try:
                prediction_data = {
                    'user': session.get('user'),
                    'prediction': prediction_result,
                    'confidence': f"{confidence:.2f}",
                    'image_name': filename,
                    'timestamp': datetime.now().isoformat(),
                    'all_probabilities': {
                        CLASS_LABELS[i]: f"{float(predictions[0][i]) * 100:.2f}"
                        for i in range(len(CLASS_LABELS))
                    }
                }
                my_database.create_document(prediction_data)
                print("✅ Prediction saved to database")
            except Exception as e:
                print(f"⚠️  Failed to save prediction: {e}")
        
        return render_template('prediction.html', 
                             prediction=prediction_result,
                             confidence=f"{confidence:.2f}",
                             image_filename=filename,
                             user=session.get('name', 'User'),
                             pred='Logout',
                             vis='hidden',
                             vis2='visible',
                             show_result=True)
    
    flash('Invalid file type. Please upload PNG, JPG, or JPEG.', 'error')
    return render_template('index.html', error='Invalid file type')

if __name__ == '__main__':
    print("="*60)
    print("DIABETIC RETINOPATHY DETECTION SYSTEM")
    print("="*60)
    
    # Check model
    if not os.path.exists(MODEL_PATH):
        print("⚠️  Model not found!")
        print("Please train the model first using:")
        print("  1. Wait for training to complete")
        print("  2. Or run: python train_model.py")
        print("="*60)
    else:
        print("✅ Model found!")
        load_model_if_exists()
    
    # Initialize Cloudant
    print("\nInitializing IBM Cloudant Database...")
    if init_cloudant():
        print("✅ Cloudant database connected")
    else:
        print("⚠️  Cloudant not configured (app will work without it)")
        print("To enable Cloudant:")
        print("  1. Create Cloudant service on IBM Cloud")
        print("  2. Update cloudant_config.py with credentials")
    
    print("\n" + "="*60)
    print("Starting Flask server...")
    print("Access at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
