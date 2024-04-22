#libraries
import streamlit as st
from streamlit_option_menu import option_menu
from ImageGen import CGI
from audiorecorder import audiorecorder
from transcribe import audio_transcription
import time
from Img2Txt import img2txt_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from transcribe import save_audio_file

#download AI genertaed images
def download_image(image):
    st.write('Generating Image download link, please wait...')
    time.sleep(5)
    image_stream = BytesIO()
    image.save(image_stream, format="JPEG")

    # Convert the BytesIO object to a base64 string
    image_base64 = base64.b64encode(image_stream.getvalue()).decode()

    # Display a link to download the image
    st.markdown(
        f'<a href="data:image/jpeg;base64,{image_base64}" download="captured_image.jpg">Click here to download image</a>',
        unsafe_allow_html=True
    )



#streamlit app
def main():
    with st.sidebar:
        navbar_menu = option_menu(
            menu_title="VisualInsights",
            options=['Image Generation', 'Image2text'],
            icons=["file-earmark-image", "file-earmark-image-fill"],
            menu_icon='menu-button',
            default_index=0,
            orientation="vertical"
        )

    #Image Generation feature
    if navbar_menu == 'Image Generation':
        header = st.title('Image Generation using `Stable Diffusion` :milky_way:')
        prompt = st.text_area('Enter Prompt')
        st.write("or Auto-record")
        filepath = "audio.wav" 
        
        #Audio record feature
        audio = audiorecorder("Click to record", "Click to stop recording")
        if len(audio) > 0:
            audio_rec = st.audio(audio.export().read())

            # Save audio to a file
            save_audio_file(audio, filepath)
            # Transcribe audio
        transcribed_text = audio_transcription(filepath)
        if st.button('Generate'):
            if len(audio) > 0:
              st.info(transcribed_text)
              generated_trans_image = CGI(transcribed_text)
              st.image(generated_trans_image, use_column_width=False)
              download_image(generated_trans_image)
            elif len(prompt) != 0:
              st.info(prompt)
              generated_image = CGI(prompt) 
              st.image(generated_image, use_column_width=False)
              download_image(generated_image)
              
            else:
              st.error('No input provided')
        if st.button("Clear Audio-Input"):
          audio_rec.empty()

     #Image2text feature     
    elif navbar_menu == 'Image2text':
        st.title('Image to text')
        uploaded_file = st.file_uploader('Upload Image', type=['PNG', 'JPEG'])
        prompt_text = st.text_area('Enter Prompt')
        st.write("or Auto-record")
        audio_filepath = "audio.wav"     

        #Audio record feature    
        audio = audiorecorder("Click to record", "Click to stop recording")
        if len(audio) > 0:
            audio_rec = st.audio(audio.export().read())
            # Save audio to a file
            save_audio_file(audio, audio_filepath)
            # Transcribe audio
        transcribed_text = audio_transcription(audio_filepath)
        if uploaded_file is not None:
          filepath = '' + uploaded_file.name
          with open(filepath, 'wb') as f:
            f.write(uploaded_file.read())
        if st.button('Generate'):
          col1, col2 = st.columns(2)
          if len(audio) > 0:
            with col1:
             st.info(transcribed_text)
             st.image(filepath)
             with col2:
              st.info(img2txt_model(filepath, transcribed_text))
          elif len(prompt_text) != 0:
            with col1:
              st.info(prompt_text)
              st.image(filepath)
            with col2:
              st.info(img2txt_model(filepath, prompt_text))
          else:
            st.error('No input provided')
        if st.button("Clear Audio-Input"):
            audio_rec.empty()
    st.markdown('<div class="footer" style="text-align: center;">Powered by Streamlit❤ and HuggingFace🤗 Space<br> Created by <a href="https://www.linkedin.com/in/clinton-odufuwa-7914b11b0/">Clinton</a></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()