from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

df = pd.read_csv("enhanced_dataset.csv")

df.columns = df.columns.str.strip()

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("Home.html")


@app.route("/search", methods=["POST"])
def search():

    subject = request.form.get("subject")
    topic = request.form.get("topic")
    level = request.form.get("level")

    input_text = subject + " " + topic + " " + level

    input_vector = vectorizer.transform([input_text])

    distances, indices = model.kneighbors(input_vector)

    resources = df.iloc[indices[0]].to_dict(orient="records")

    for resource in resources:

        s = resource["Subject"]
        t = resource["Topic"]
        l = resource["Level"]
        rtype = resource["ResourceType"]

        if rtype == "Video":
            resource["Link"] = f"https://www.youtube.com/results?search_query={s}+{t}+{l}"

        elif rtype == "Article":
            resource["Link"] = f"https://en.wikipedia.org/wiki/{t}"

        elif rtype == "Worksheet":
            resource["Link"] = f"https://www.google.com/search?q={s}+{t}+worksheet+pdf"

        elif rtype == "Quiz":
            resource["Link"] = f"https://www.google.com/search?q={s}+{t}+quiz"

    return render_template(
        "Results.html",
        resources=resources,
        subject=subject,
        topic=topic,
        level=level
    )
if __name__ == "__main__":
    app.run(debug=True)