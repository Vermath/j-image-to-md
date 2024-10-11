import streamlit as st
import pandas as pd
import base64
import os
from openai import OpenAI
import markdown
import streamlit.components.v1 as components
import json

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

def generate_website_code(recipes, website_name, theme):
    # Prepare the prompt for code generation
    prompt = f"""
You are a web developer. Create a simple, responsive HTML website named "{website_name}". The website should display the following recipes in a user-friendly format. Include appropriate CSS styles (consider dark mode if necessary). Each recipe should be on its own page, with a main index page listing all recipes with links to their pages.

Recipes:
{json.dumps(recipes)}

Provide the complete HTML, CSS, and JavaScript code necessary for the website.
"""

    # Generate the website code using OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates website code."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
    )

    website_code = response.choices[0].message.content.strip()
    return website_code

def main():
    st.title("Handwritten Recipe Transcriber")
    st.write("Upload images of handwritten recipes to generate a concept website based on them.")

    # Detect if dark mode is being used
    theme = st.get_option("theme.base")
    if theme == "dark":
        text_color = "white"
        bg_color = "#0e1117"
    else:
        text_color = "black"
        bg_color = "white"

    # Ask for website name
    website_name = st.text_input("Enter a name for your website:", "My Recipe Book")

    uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

    if uploaded_files and website_name:
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
            st.write("## Transcribed Recipes")
            df = pd.DataFrame(results)
            st.dataframe(df)

            # Generate the website code
            with st.spinner("Generating your website..."):
                try:
                    recipes = []
                    for recipe in results:
                        title = recipe["Transcribed Text"].split('\n')[0].replace('#', '').strip()
                        content = recipe["Transcribed Text"]
                        recipes.append({"title": title, "content": content})
                    website_code = generate_website_code(recipes, website_name, theme)
                    st.success("Website generated successfully!")
                except Exception as e:
                    st.error(f"Error generating website: {e}")
                    return

            # Render the website within the app
            st.write("## Your Generated Website")
            components.v1.html(website_code, height=800, scrolling=True)

            # Provide option to download the website code
            b64_code = base64.b64encode(website_code.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64_code}" download="{website_name.replace(" ", "_")}.html">Download Website Code</a>'
            st.markdown(href, unsafe_allow_html=True)

    # Adjust styles for dark mode
    st.markdown(f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .stApp {{
        background-color: {bg_color};
    }}
    .css-1d391kg {{
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
