import streamlit as st
import pandas as pd
import base64
import os
from openai import OpenAI

# Initialize OpenAI client
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

def transcribe_image(image_data, image_name):
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

def main():
    st.title("Handwritten Recipe Transcriber")
    st.write("Upload images of handwritten recipes to get transcribed markdown and download a CSV file.")

    uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    
    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            image_name = uploaded_file.name
            image_data = uploaded_file.read()
            with st.spinner(f"Transcribing {image_name}..."):
                try:
                    transcription = transcribe_image(image_data, image_name)
                    results.append({"Image Name": image_name, "Transcribed Text": transcription})
                    st.success(f"Transcribed {image_name}")
                except Exception as e:
                    st.error(f"Error transcribing {image_name}: {e}")

        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)

            # Convert dataframe to CSV
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # B64 encode
            href = f'<a href="data:file/csv;base64,{b64}" download="transcriptions.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
