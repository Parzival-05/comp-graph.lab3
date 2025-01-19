import logging
import os

import numpy as np
from flask import Flask, render_template, request, send_from_directory

from src.infer import LABELS, infer_by_path

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    filemode="w",
    format="%(asctime)s: %(name)s - [%(levelname)s]: %(message)s",
)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    error = None
    results = None
    predicted_label = None
    image_filename = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file selected"
            return render_template("index.html", error=error)

        file = request.files["file"]

        if file.filename == "":
            error = "No file selected"
            return render_template("index.html", error=error)

        if file and allowed_file(file.filename):
            for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logging.error(f"Error deleting old file: {e}")

            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)  # type: ignore
            try:
                file.save(filepath)
                probabilities = infer_by_path(filepath)

                predicted_class_index = int(np.argmax(probabilities))
                predicted_label = LABELS[predicted_class_index]
                results_dict = {
                    LABELS[i]: f"{p:.4f}" for i, p in enumerate(probabilities)
                }

                results = results_dict
                image_filename = filename

            except ValueError as e:
                error = f"Error processing the image: {e}"
                logging.error(e, exc_info=True)
            except Exception as e:
                error = f"An unexpected error occurred: {e}"
                logging.error(e, exc_info=True)
        else:
            error = "Invalid file format. Only png, jpg, jpeg are allowed."

    return render_template(
        "index.html",
        error=error,
        results=results,
        predicted_label=predicted_label,
        image_filename=image_filename,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run()
