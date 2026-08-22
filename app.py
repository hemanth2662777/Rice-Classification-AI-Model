import os
import sqlite3

import numpy as np

from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from werkzeug.utils import secure_filename

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


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "grainpalette-ai-development-secret-key"
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

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
    connection.close()

    print("✅ Database initialized.")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("==============================================")
print("       GRAINPALETTE AI - STARTING")
print("==============================================")

print(f"Model path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"❌ Trained model not found: {MODEL_PATH}"
    )


try:

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    print("✅ Trained rice model loaded successfully.")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

except Exception as e:

    print("❌ Model loading failed.")
    print(e)

    raise


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_rice(image_path):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (224, 224)
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = image_array / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            np.max(predictions[0]) * 100
        )

        if predicted_index >= len(CLASS_NAMES):

            raise ValueError(
                "Model returned an unexpected class index."
            )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        return (
            predicted_class,
            round(confidence, 2)
        )

    except Exception as e:

        print(
            "❌ Prediction error:",
            e
        )

        raise


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(function):

    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            return redirect(
                url_for("login")
            )

        return function(
            *args,
            **kwargs
        )

    wrapper.__name__ = function.__name__

    return wrapper


# ============================================================
# HOME
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

        if (
            uploaded_file is None
            or uploaded_file.filename == ""
        ):

            error = "Please select a rice image."

        elif not allowed_file(
            uploaded_file.filename
        ):

            error = (
                "Invalid file type. "
                "Please upload JPG, JPEG or PNG."
            )

        else:

            try:

                original_filename = secure_filename(
                    uploaded_file.filename
                )

                # Make filename unique
                filename = (
                    f"{session['user_id']}_"
                    f"{original_filename}"
                )

                file_path = os.path.join(
                    UPLOAD_DIR,
                    filename
                )

                uploaded_file.save(
                    file_path
                )

                prediction, confidence = predict_rice(
                    file_path
                )

                image_path = url_for(
                    "static",
                    filename=f"uploads/{filename}"
                )

            except Exception as e:

                print(
                    "❌ Upload/prediction error:",
                    e
                )

                error = (
                    "Unable to analyze the image. "
                    "Please try another image."
                )

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

            connection = get_db_connection()

            user = connection.execute(
                """
                SELECT *
                FROM users
                WHERE username = ?
                """,
                (username,)
            ).fetchone()

            connection.close()

            if (
                user
                and check_password_hash(
                    user["password"],
                    password
                )
            ):

                session["user_id"] = user["id"]
                session["username"] = user["username"]

                return redirect(
                    url_for("home")
                )

            error = (
                "Invalid username or password."
            )

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

                connection.close()

                error = (
                    "Username already exists. "
                    "Please choose another username."
                )

            else:

                password_hash = generate_password_hash(
                    password
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
                connection.close()

                return redirect(
                    url_for("login")
                )

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
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    init_database()

    print("==============================================")
    print("GrainPalette AI is ready.")
    print("Open: http://127.0.0.1:7860")
    print("==============================================")

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )