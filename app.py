import streamlit as st
import pandas as pd
import base64
import os
from openai import OpenAI
import markdown
from pathlib import Path

# Initialize OpenAI client
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

def transcribe_image(image_data):
    # Encode image to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')
    # Prepare the API request
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe the following handwritten recipe into markdown format."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=1500,
    )
    # Extract the transcribed text
    transcribed_text = response.choices[0].message.content.strip()
    return transcribed_text

def generate_html(recipes):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>My Recipe Book</title>
        <style>
            body {font-family: Arial, sans-serif; margin: 20px;}
            .recipe {margin-bottom: 50px;}
            h1 {text-align: center;}
        </style>
    </head>
    <body>
        <h1>My Recipe Book</h1>
    """
    for recipe in recipes:
        # Convert markdown to HTML
        html_recipe = markdown.markdown(recipe["Transcribed Text"])
        html_content += f'<div class="recipe">{html_recipe}</div>\n'
    html_content += """
    </body>
    </html>
    """
    return html_content

def main():
    st.title("Handwritten Recipe Transcriber")
    st.write("Upload images of handwritten recipes to get beautifully rendered markdown and view your recipe website.")

    uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            image_name = uploaded_file.name
            image_data = uploaded_file.read()
            with st.spinner(f"Transcribing {image_name}..."):
                try:
                    transcription = transcribe_image(image_data)
                    results.append({"Image Name": image_name, "Transcribed Text": transcription})
                    st.success(f"Transcribed {image_name}")
                except Exception as e:
                    st.error(f"Error transcribing {image_name}: {e}")

        if results:
            df = pd.DataFrame(results)
            st.write("## Transcribed Recipes")
            st.dataframe(df)

            # Generate and display the website
            st.write("## Recipe Website")
            html_content = generate_html(results)
            # Render the HTML content in an iframe
            st.components.v1.html(html_content, height=800, scrolling=True)

            # Provide option to download the HTML file
            b64_html = base64.b64encode(html_content.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64_html}" download="recipe_book.html">Download Recipe Website as HTML</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Provide CSV download
            csv = df.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="transcriptions.csv">Download CSV File</a>'
            st.markdown(href_csv, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
