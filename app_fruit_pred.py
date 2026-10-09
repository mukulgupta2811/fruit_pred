from flask import Flask, request, render_template_string
import cv2
import numpy as np
import fruit_pred

app = Flask(__name__)


HTML = """
<!DOCTYPE html>
<html>

<head>

    <title>Fruit Detection</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #e8f5e9, #ffffff);
            min-height: 100vh;

            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: 600px;
            background: white;
            padding: 40px;
            border-radius: 25px;
            text-align: center;

            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        }

        h1 {
            color: #2e7d32;
            font-size: 42px;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #666;
            font-size: 17px;
            margin-bottom: 30px;
        }

        .upload-box {
            border: 2px dashed #66bb6a;
            padding: 30px;
            border-radius: 18px;
            background: #f8fff8;
        }

        input[type="file"] {
            margin: 15px;
            font-size: 15px;
        }

        button {
            background: #43a047;
            color: white;
            border: none;

            padding: 13px 35px;

            border-radius: 10px;

            font-size: 17px;

            cursor: pointer;
        }

        button:hover {
            background: #2e7d32;
        }

        .result {
            margin-top: 30px;
            padding: 20px;
            background: #e8f5e9;
            border-radius: 15px;
        }

        .fruit {
            font-size: 30px;
            font-weight: bold;
            color: #2e7d32;
        }

        .confidence {
            margin-top: 10px;
            font-size: 18px;
            color: #555;
        }

        .error {
            margin-top: 20px;
            color: red;
            font-weight: bold;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>🍎 Fruit Detection</h1>

    <p class="subtitle">
        Upload a fruit image and let CNN predict it
    </p>


    <form method="POST" enctype="multipart/form-data">

        <div class="upload-box">

            <input
                type="file"
                name="image"
                accept="image/*"
                required
            >

            <br>

            <button type="submit">
                Predict Fruit
            </button>

        </div>

    </form>


    {% if result %}

    <div class="result">

        <div class="fruit">
            🍓 {{ result }}
        </div>

        <div class="confidence">
            Confidence: {{ confidence }}%
        </div>

    </div>

    {% endif %}


    {% if error %}

    <div class="error">
        {{ error }}
    </div>

    {% endif %}

</div>

</body>

</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    error = None

    if request.method == "POST":

        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please select an image."

            return render_template_string(
                HTML,
                result=result,
                confidence=confidence,
                error=error
            )


        # Uploaded image ko read karna
        file_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if image is None:

            error = "Image read nahi ho paayi."

            return render_template_string(
                HTML,
                result=result,
                confidence=confidence,
                error=error
            )


        # Image ko same size mein convert karna
        image = cv2.resize(
            image,
            (128, 128)
        )

        # Same preprocessing as training
        image = image / 255.0

        # Batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )


        # Prediction
        pred = fruit_pred.model.predict(image)

        prediction = np.argmax(pred[0])

        confidence = np.max(pred[0]) * 100

        # Fruit ka naam
        result = fruit_pred.classes[prediction]

        confidence = round(float(confidence), 2)


    return render_template_string(
        HTML,
        result=result,
        confidence=confidence,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)