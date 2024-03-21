import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, flash, redirect, url_for
import cv2
import numpy as np

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def processImage(filename, operation):
    print(f"The Operation is {operation} and the filename is {filename}")
    img = cv2.imread(f"uploads/{filename}")
    match operation:
        case "grayscale":
            new = f"static/{filename}"
            imgProcessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(new, imgProcessed)
            return new
        case "resize50":
            new = f"static/{filename}"
            imgProcessed = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            cv2.imwrite(new, imgProcessed)
            return new
        case "resize30":
            new = f"static/{filename}"
            imgProcessed = cv2.resize(img, (int(img.shape[1]*0.3), int(img.shape[0]*0.3)))
            cv2.imwrite(new, imgProcessed)
            return new
        case "resize80":
            new = f"static/{filename}"
            imgProcessed = cv2.resize(img,(int(img.shape[1]*0.8), int(img.shape[0]*0.8)))
            cv2.imwrite(new, imgProcessed)
            return new
        case "mirror":
            new = f"static/{filename}"
            imgProcessed = cv2.flip(img, 1)
            cv2.imwrite(new, imgProcessed)
            return new
        case "squarecrop":
            new = f"static/{filename}"
            min_dimension = min(img.shape[0], img.shape[1])
            x1 = (img.shape[1] - min_dimension) // 2
            x2 = x1 + min_dimension
            y1 = (img.shape[0] - min_dimension) // 2
            y2 = y1 + min_dimension
            imgProcessed = img[y1:y2, x1:x2]
            cv2.imwrite(new, imgProcessed)
            return new
        case "rotate90":
            new = f"static/{filename}"  # New filename for rotated image
            img_processed = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate image by 90 degrees clockwise
            cv2.imwrite(new, img_processed)  # Save rotated image
            return new  # Return the path to the rotated image
        case "rotate180":
            new = f"static/{filename}"  # New filename for rotated image
            img_processed = cv2.rotate(img, cv2.ROTATE_180)  # Rotate image by 180 degrees
            cv2.imwrite(new, img_processed)  # Save rotated image
            return new  # Return the path to the rotated image
        case "rotate270":
            new = f"static/{filename}"  # New filename for rotated image
            img_processed = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate image by 180 degrees
            cv2.imwrite(new, img_processed)  # Save rotated image
            return new  # Return the path to the rotated image
        case "contrast_0.1":
            new = f"static/{filename}"
            img_processed = np.array(255*(img / 255) ** 0.1, dtype = 'uint8')
            cv2.imwrite(new, img_processed)
            return new
        case "contrast_0.5":
            new = f"static/{filename}"
            img_processed = np.array(255*(img / 255) ** 0.5, dtype = 'uint8')
            cv2.imwrite(new, img_processed)
            return new
        case "contrast_1.2":
            new = f"static/{filename}"
            img_processed = np.array(255*(img / 255) ** 1.2, dtype = 'uint8')
            cv2.imwrite(new, img_processed)
            return new
        case "contrast_2.2":
            new = f"static/{filename}"
            img_processed = np.array(255*(img / 255) ** 2.2, dtype = 'uint8')
            cv2.imwrite(new, img_processed)
            return new
        case "Face_Blur":
            new = f"static/{filename}"
            # Converting BGR image into a RGB image
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Load the Haar Cascade classifier
            face_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
            face_data = face_cascade.detectMultiScale(image, 1.3, 5)
            # Draw rectangle around the faces which is our region of interest (ROI) 
            for (x, y, w, h) in face_data: 
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                roi = image[y:y+h, x:x+w] 
                # applying a gaussian blur over this new rectangle area 
                roi = cv2.GaussianBlur(roi, (23, 23), 30) 
                # impose this blurred image on original image to get final image 
                image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
            cv2.imwrite(new, image)
            return new
        case "Image_Blur":
            new = f"static/{filename}"
            # ksize 
            ksize = (15, 15) 
            # Using cv2.blur() method  
            img_processed = cv2.blur(img, ksize)
            cv2.imwrite(new, img_processed)
            return new


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/edit", methods=["GET", "POST"])
def edit():
    if request.method == 'POST':
        operation = request.form.get("operation")
        if 'file' not in request.files:
            flash('No file part')
            return "error"
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return "Error: No Selected File"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new = processImage(filename, operation)
            flash(
                f"Your image has been processed and is available <a target='_blank' href='/{new}'>here</a>")

            return render_template("index.html", new = new)


app.run(debug=True)
