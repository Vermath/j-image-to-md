import streamlit as st
import pandas as pd
import base64
import os
from openai import OpenAI
import streamlit.components.v1 as components
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize OpenAI client
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

def transcribe_image(image_data, image_name):
    """
    Transcribe a single image of a handwritten recipe into markdown format using OpenAI's GPT-4o-mini.

    Args:
        image_data (bytes): The binary data of the image.
        image_name (str): The name of the image file.

    Returns:
        dict: A dictionary containing 'Image Name' and 'Transcribed Text'.
    """
    try:
        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare the API request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Transcribe the following handwritten recipe into markdown format without any commentary. Image Name: {image_name}."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=16000  # Adjust as needed, up to the model's limit 
        )
        
        # Extract the transcribed text
        transcribed_text = response.choices[0].message.content.strip()
        return {"Image Name": image_name, "Transcribed Text": transcribed_text}
    
    except Exception as e:
        st.error(f"❌ Error transcribing {image_name}: {e}")
        return {"Image Name": image_name, "Transcribed Text": ""}

def generate_website_code(recipes, website_name):
    """
    Generate a complete multi-page HTML website with embedded CSS and JavaScript based on the transcribed recipes using OpenAI's o1-mini.

    Args:
        recipes (list of dict): Each dict contains 'title' and 'content'.
        website_name (str): The name of the website.

    Returns:
        str: The complete HTML code for the website.
    """
    try:
        # Prepare the prompt for code generation
        prompt = f"""
Create a complete, responsive multi-page HTML website named "{website_name}". The website should display the following recipes in a user-friendly format with proper navigation. Include embedded CSS styles that support both light and dark modes based on the user's system preferences.

Recipes:
{json.dumps(recipes, indent=2)}

Requirements:
- A navigation bar with the website name.
- A main index page listing all recipes with links to their respective pages.
- Each recipe should have its own page displaying its title and content.
- Responsive design for mobile and desktop.
- Dark mode support using CSS media queries.
- Embedded CSS and JavaScript within the HTML files (no separate files).
- No explanations or additional text; only provide the complete HTML code.
"""

        # Make the API call to o1-mini without system prompts
        response = client.chat.completions.create(
            model="o1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=64000  # Adjust as needed, up to the model's limit 
        )

        website_code = response.choices[0].message.content.strip()
        
        # Extract code within triple backticks if present
        code_match = re.search(r"```html\s*(.*?)\s*```", website_code, re.DOTALL | re.IGNORECASE)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # If no backticks, assume the entire response is code
            code = website_code

        # Validate that the code contains essential HTML structure
        if not re.search(r"<!DOCTYPE html>", code, re.IGNORECASE):
            st.warning("⚠️ The generated website code might be incomplete or improperly formatted.")
        
        return code

    except Exception as e:
        st.error(f"❌ Error generating website: {e}")
        return ""

def main():
    st.set_page_config(page_title="📸 Handwritten Recipe Transcriber", layout="wide")
    st.title("📸 Handwritten Recipe Transcriber")
    st.write("Upload images of handwritten recipes to generate a concept website based on them.")

    # Detect if dark mode is being used
    theme = st.get_option("theme.base")
    if theme == "dark":
        text_color = "#FFFFFF"
        bg_color = "#0e1117"
    else:
        text_color = "#000000"
        bg_color = "#FFFFFF"

    # Apply dynamic CSS based on the theme
    st.markdown(f"""
    <style>
    .main {{
        background-color: {bg_color};
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Ask for website name
    website_name = st.text_input("🌐 Enter a name for your website:", "My Recipe Book")

    uploaded_files = st.file_uploader("📂 Choose image files", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

    if uploaded_files and website_name:
        images = []
        for uploaded_file in uploaded_files:
            images.append({
                "image_name": uploaded_file.name,
                "image_data": uploaded_file.read()
            })
        
        st.markdown("### 📄 Transcribed Recipes")
        df = pd.DataFrame(columns=["Image Name", "Transcribed Text"])

        with st.spinner("📝 Transcribing uploaded images..."):
            # Use ThreadPoolExecutor to parallelize transcription
            with ThreadPoolExecutor() as executor:
                # Submit all transcription tasks
                future_to_image = {executor.submit(transcribe_image, img['image_data'], img['image_name']): img for img in images}
                results = []
                for future in as_completed(future_to_image):
                    res = future.result()
                    results.append(res)
        
        # Create DataFrame from results
        df = pd.DataFrame(results)
        st.dataframe(df)

        # Provide option to download the CSV file
        csv = df.to_csv(index=False)
        b64_csv = base64.b64encode(csv.encode()).decode()  # B64 encode
        href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="transcriptions.csv">📥 Download CSV File</a>'
        st.markdown(href_csv, unsafe_allow_html=True)

        # Prepare recipes for website generation
        recipes = []
        for index, row in df.iterrows():
            transcription = row["Transcribed Text"]
            # Extract title from markdown (assuming the first line is the title)
            lines = transcription.split('\n')
            if lines and lines[0].startswith('#'):
                title = lines[0].replace('#', '').strip()
                content = '\n'.join(lines[1:]).strip()
            else:
                title = "Untitled Recipe"
                content = transcription
            recipes.append({
                "title": title,
                "content": content
            })
        
        with st.spinner("💻 Generating your website..."):
            website_code = generate_website_code(recipes, website_name)
            if website_code:
                st.success("🎉 Website generated successfully!")

                # Render the website within the app in its own independent frame using srcdoc
                st.markdown("## 🌐 Your Generated Website")
                components.html(
                    f"""
                    <iframe srcdoc='{website_code}' width="100%" height="800px" frameborder="0" allowfullscreen></iframe>
                    """,
                    height=800,
                    scrolling=True
                )
                
                # Provide option to download the website code
                b64_code = base64.b64encode(website_code.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64_code}" download="{website_name.replace(" ", "_")}.html">📥 Download Website Code</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("❌ Failed to generate website code.")

    elif uploaded_files and not website_name:
        st.warning("⚠️ Please enter a name for your website.")
    
    else:
        st.info("📌 Please upload images of handwritten recipes and enter a website name to get started.")

# Run the app
if __name__ == "__main__":
    main()
