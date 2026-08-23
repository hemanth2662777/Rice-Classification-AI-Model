import os
import sqlite3
import uuid
from functools import wraps

# ============================================================
# TENSORFLOW CPU / MEMORY CONFIGURATION
# ============================================================

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "rice_model.h5"
)

DATABASE_DIR = os.path.join(
    BASE_DIR,
    "database"
)

DATABASE_PATH = os.path.join(
    DATABASE_DIR,
    "users.db"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

CLASS_NAMES = [
    "Arborio",
    "Basmati",
    "Ipsala",
    "Jasmine",
    "Karacadag"
]

IMAGE_SIZE = (224, 224)

MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "grainpalette-ai-development-secret-key"
)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

PORT = int(
    os.environ.get(
        "PORT",
        7860
    )
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    DATABASE_DIR,
    exist_ok=True
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = None

    try:

        connection = get_db_connection()

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT,
                last_name TEXT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password TEXT NOT NULL
            )
            """
        )

        connection.commit()

        print("✅ Database initialized.")

    except Exception as e:

        print(
            "❌ Database initialization error:",
            repr(e)
        )

        raise

    finally:

        if connection is not None:
            connection.close()


init_database()


# ============================================================
# STARTUP
# ============================================================

print("==============================================")
print("       GRAINPALETTE AI - STARTING")
print("==============================================")

print(
    f"Model path: {MODEL_PATH}"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.isfile(MODEL_PATH):

    raise FileNotFoundError(
        f"❌ Trained model not found: {MODEL_PATH}"
    )


# ============================================================
# TENSORFLOW THREAD CONFIGURATION
# ============================================================

try:

    tf.config.threading.set_intra_op_parallelism_threads(1)

    tf.config.threading.set_inter_op_parallelism_threads(1)

    print(
        "✅ TensorFlow CPU threading configured."
    )

except RuntimeError as e:

    print(
        "⚠️ TensorFlow threading configuration skipped:",
        e
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "✅ Trained rice model loaded successfully."
    )

    print(
        "Input shape:",
        model.input_shape
    )

    print(
        "Output shape:",
        model.output_shape
    )

except Exception as e:

    print(
        "❌ Model loading failed:",
        repr(e)
    )

    raise


# ============================================================
# MODEL VALIDATION
# ============================================================

try:

    output_shape = model.output_shape

    if (
        output_shape is None
        or len(output_shape) != 2
        or output_shape[-1] != len(CLASS_NAMES)
    ):

        raise ValueError(
            "Model output does not match the expected "
            f"{len(CLASS_NAMES)} rice classes."
        )

    print(
        "✅ Model class count validated."
    )

except Exception as e:

    print(
        "❌ Model validation failed:",
        repr(e)
    )

    raise


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# RICE PREDICTION
# ============================================================

def predict_rice(image_path):

    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        image_array = image_array / 255.0

        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        print(
            "Image input shape:",
            image_array.shape
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )

        predictions = np.asarray(
            predictions
        )

        print(
            "Raw predictions:",
            predictions
        )

        # ----------------------------------------------------
        # VALIDATE OUTPUT
        # ----------------------------------------------------

        if (
            predictions.ndim != 2
            or predictions.shape[0] != 1
            or predictions.shape[1] != len(CLASS_NAMES)
        ):

            raise ValueError(
                "Unexpected model prediction shape: "
                f"{predictions.shape}"
            )

        # ----------------------------------------------------
        # GET CLASS
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )

        confidence = float(
            predictions[0][predicted_index]
        )

        # ----------------------------------------------------
        # SAFETY: HANDLE LOGITS
        # ----------------------------------------------------

        # If the model outputs logits instead of probabilities,
        # convert them to probabilities.

        if confidence < 0 or confidence > 1:

            exp_predictions = np.exp(
                predictions[0]
                - np.max(predictions[0])
            )

            probabilities = (
                exp_predictions
                / np.sum(exp_predictions)
            )

            confidence = float(
                probabilities[predicted_index]
            )

        # ----------------------------------------------------
        # CLASS NAME
        # ----------------------------------------------------

        if not (
            0 <= predicted_index < len(CLASS_NAMES)
        ):

            raise ValueError(
                "Invalid predicted class index."
            )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence_percent = round(
            confidence * 100,
            2
        )

        print(
            f"✅ Prediction: {predicted_class}"
        )

        print(
            f"✅ Confidence: {confidence_percent}%"
        )

        return (
            predicted_class,
            confidence_percent
        )

    except Exception as e:

        print(
            "❌ Prediction error:",
            repr(e)
        )

        raise


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            return redirect(
                url_for("login")
            )

        return function(
            *args,
            **kwargs
        )

    return wrapper


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify(
        {
            "status": "healthy",
            "service": "GrainPalette AI",
            "model_loaded": model is not None,
            "classes": CLASS_NAMES
        }
    ), 200


# ============================================================
# HOME / RICE CLASSIFICATION
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
@login_required
def home():

    prediction = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":

        uploaded_file = request.files.get(
            "file"
        )

        # ----------------------------------------------------
        # FILE VALIDATION
        # ----------------------------------------------------

        if (
            uploaded_file is None
            or not uploaded_file.filename
        ):

            error = (
                "Please select a rice image."
            )

        elif not allowed_file(
            uploaded_file.filename
        ):

            error = (
                "Invalid file type. "
                "Please upload JPG, JPEG or PNG."
            )

        else:

            file_path = None

            try:

                # ------------------------------------------------
                # SECURE ORIGINAL FILENAME
                # ------------------------------------------------

                original_filename = secure_filename(
                    uploaded_file.filename
                )

                if not original_filename:

                    raise ValueError(
                        "Invalid filename."
                    )

                # ------------------------------------------------
                # GET EXTENSION
                # ------------------------------------------------

                extension = os.path.splitext(
                    original_filename
                )[1].lower()

                # ------------------------------------------------
                # CREATE UNIQUE FILENAME
                # ------------------------------------------------

                filename = (
                    f"{session['user_id']}_"
                    f"{uuid.uuid4().hex}"
                    f"{extension}"
                )

                # ------------------------------------------------
                # ABSOLUTE FILE PATH
                # ------------------------------------------------

                file_path = os.path.join(
                    UPLOAD_DIR,
                    filename
                )

                # ------------------------------------------------
                # SAVE UPLOADED IMAGE
                # ------------------------------------------------

                uploaded_file.save(
                    file_path
                )

                print(
                    "=============================================="
                )

                print(
                    f"📷 Image uploaded: {filename}"
                )

                print(
                    f"📁 Saved at: {file_path}"
                )

                print(
                    f"📁 Exists: {os.path.exists(file_path)}"
                )

                # ------------------------------------------------
                # PREDICT
                # ------------------------------------------------

                prediction, confidence = predict_rice(
                    file_path
                )

                # ------------------------------------------------
                # VERIFY IMAGE STILL EXISTS
                # ------------------------------------------------

                if not os.path.isfile(file_path):

                    raise FileNotFoundError(
                        "Uploaded image disappeared before "
                        "result rendering."
                    )

                # ------------------------------------------------
                # CREATE STATIC URL
                # ------------------------------------------------

                image_path = url_for(
                    "static",
                    filename=f"uploads/{filename}",
                    _external=False
                )

                print(
                    f"🖼️ Image URL: {image_path}"
                )

                print(
                    f"🖼️ Browser file exists: "
                    f"{os.path.isfile(file_path)}"
                )

                print(
                    "=============================================="
                )

            except Exception as e:

                print(
                    "❌ Upload/prediction error:",
                    repr(e)
                )

                prediction = None
                confidence = None
                image_path = None

                error = (
                    "Unable to analyze the image. "
                    "Please try another rice image."
                )

            # ====================================================
            # IMPORTANT
            # ====================================================
            #
            # DO NOT DELETE file_path HERE.
            #
            # The HTML needs the uploaded image after the POST
            # request finishes.
            #
            # The old code deleted it using os.remove().
            #
            # ====================================================

    return render_template(
        "home.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if "user_id" in session:

        return redirect(
            url_for("home")
        )

    error = None

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not username or not password:

            error = (
                "Please enter username and password."
            )

        else:

            connection = None

            try:

                connection = get_db_connection()

                user = connection.execute(
                    """
                    SELECT *
                    FROM users
                    WHERE username = ?
                    """,
                    (username,)
                ).fetchone()

                if (
                    user
                    and check_password_hash(
                        user["password"],
                        password
                    )
                ):

                    session["user_id"] = user["id"]

                    session["username"] = (
                        user["username"]
                    )

                    return redirect(
                        url_for("home")
                    )

                error = (
                    "Invalid username or password."
                )

            except Exception as e:

                print(
                    "❌ Login database error:",
                    repr(e)
                )

                error = (
                    "Unable to process login. "
                    "Please try again."
                )

            finally:

                if connection is not None:
                    connection.close()

    return render_template(
        "login.html",
        error=error
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if "user_id" in session:

        return redirect(
            url_for("home")
        )

    error = None

    if request.method == "POST":

        first_name = request.form.get(
            "first_name",
            ""
        ).strip()

        last_name = request.form.get(
            "last_name",
            ""
        ).strip()

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        terms = request.form.get(
            "terms"
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not username:

            error = "Username is required."

        elif not password:

            error = "Password is required."

        elif len(password) < 6:

            error = (
                "Password must contain "
                "at least 6 characters."
            )

        elif password != confirm_password:

            error = "Passwords do not match."

        elif not terms:

            error = (
                "Please accept the Terms of Service "
                "and Privacy Policy."
            )

        else:

            connection = None

            try:

                connection = get_db_connection()

                existing_user = connection.execute(
                    """
                    SELECT id
                    FROM users
                    WHERE username = ?
                    """,
                    (username,)
                ).fetchone()

                if existing_user:

                    error = (
                        "Username already exists. "
                        "Please choose another username."
                    )

                else:

                    password_hash = (
                        generate_password_hash(
                            password
                        )
                    )

                    connection.execute(
                        """
                        INSERT INTO users
                        (
                            first_name,
                            last_name,
                            username,
                            email,
                            password
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            first_name,
                            last_name,
                            username,
                            email,
                            password_hash
                        )
                    )

                    connection.commit()

                    print(
                        f"✅ New user registered: {username}"
                    )

                    return redirect(
                        url_for("login")
                    )

            except sqlite3.IntegrityError:

                if connection:
                    connection.rollback()

                error = (
                    "Username already exists."
                )

            except Exception as e:

                if connection:
                    connection.rollback()

                print(
                    "❌ Registration database error:",
                    repr(e)
                )

                error = (
                    "Unable to create account. "
                    "Please try again."
                )

            finally:

                if connection is not None:
                    connection.close()

    return render_template(
        "register.html",
        error=error
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    print("==============================================")
    print("       GRAINPALETTE AI - READY")
    print("==============================================")

    print(
        f"Running on port: {PORT}"
    )

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=False
    )