import streamlit as st
from audio_recorder_streamlit import audio_recorder
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import docx2txt
import PyPDF2
import io
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pydub import AudioSegment
import subprocess
import tempfile
import os

def check_ffmpeg():
    """Check if ffmpeg is installed in the system"""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def initialize_groq_client():
    """Initialize and return Groq client"""
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def transcribe_audio(client, file):
    """Transcribe audio file using Groq's API"""
    transcription = client.audio.transcriptions.create(
        file=file,
        model="whisper-large-v3",
        language="en",
        temperature=0.2
    )
    return transcription.text

def initialize_language_model():
    """Initialize and return the language model"""
    return ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"],model='gpt-4o', temperature=0.3)

def create_analysis_chain(llm):
    """Create and return the analysis chain for processing transcripts"""
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant tasked with generating comprehensive minutes of the meetings. Make sure you include all the necessary details. Do not miss any important information."),
        ("human", """Based on the following meeting transcript and information, please generate the following:

            Transcript: {transcript}

            Please provide the following information:
            1. Minutes of the meeting
            2. What was discussed
            3. A brief summary
            4. The meeting agenda (inferred from the discussion)
            5. Action items (things to be done)

            Leave space in the start for the following:
            - Date:
            - Location:
            - Attendees:
            """)
    ])

    chain = (
        {"transcript": lambda _: RunnablePassthrough()}
        | analysis_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def generate_meeting_minutes(chain, content):
    """Generate meeting minutes from content using the provided chain"""
    return chain.invoke(content)

def read_docx(file):
    """Read and return content from a DOCX file"""
    return docx2txt.process(file)

def read_pdf(file):
    """Read and return content from a PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def save_as_docx(content):
    """Save content as a DOCX file and return the file path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
        doc = Document()
        doc.add_paragraph(content)
        doc.save(tmp.name)
        return tmp.name

def save_as_pdf(content):
    """Save content as a PDF file and return the file path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        c = canvas.Canvas(tmp.name, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica", 12)

        lines = content.split('\n')
        y = height - 40

        for line in lines:
            while len(line) > 0:
                if y < 40:
                    c.showPage()
                    y = height - 40
                line_part = line[:80]
                c.drawString(40, y, line_part)
                y -= 15
                line = line[80:]

        c.save()
        return tmp.name

def convert_m4a_to_mp3(input_file):
    """Convert M4A file to MP3 format"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_m4a:
        temp_m4a.write(input_file.getvalue())
        temp_m4a_path = temp_m4a.name

    output_mp3 = temp_m4a_path.replace('.m4a', '.mp3')

    audio = AudioSegment.from_file(temp_m4a_path, format="m4a")
    audio.export(output_mp3, format="mp3")

    os.unlink(temp_m4a_path)  # Remove the temporary M4A file

    return output_mp3

def save_audio_file(audio_bytes):
    """Save audio bytes to a temporary MP3 file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tmp.write(audio_bytes)
        return tmp.name

def process_audio_content(audio_file):
    """Process audio file and generate meeting minutes"""
    groq_client = initialize_groq_client()
    content = transcribe_audio(groq_client, audio_file)

    llm = initialize_language_model()
    chain = create_analysis_chain(llm)
    minutes = generate_meeting_minutes(chain, content)

    return content, minutes

def create_download_buttons(minutes):
    """Create and display download buttons for DOCX and PDF formats"""
    docx_file = save_as_docx(minutes)
    pdf_file = save_as_pdf(minutes)

    col1, col2 = st.columns(2)

    with open(docx_file, "rb") as file:
        col1.download_button(
            label="Download DOCX",
            data=file,
            file_name="meeting_minutes.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    with open(pdf_file, "rb") as file:
        col2.download_button(
            label="Download PDF",
            data=file,
            file_name="meeting_minutes.pdf",
            mime="application/pdf"
        )

    # Clean up temporary files
    os.unlink(docx_file)
    os.unlink(pdf_file)

def display_results(content, minutes):
    """Display the transcribed content and generated minutes"""
    st.subheader("Content")
    st.text_area("Full Content", content, height=200)

    st.subheader("Meeting Minutes")
    st.markdown(minutes)

    create_download_buttons(minutes)

def handle_uploaded_file(uploaded_file):
    """Handle uploaded file processing"""
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension in ['mp3', 'wav', 'm4a']:
        if file_extension == 'm4a':
            with st.spinner("Converting M4A to MP3..."):
                mp3_file = convert_m4a_to_mp3(uploaded_file)
            st.audio(mp3_file, format='audio/mp3')

            with open(mp3_file, 'rb') as audio_file:
                content, minutes = process_audio_content(audio_file)
            os.unlink(mp3_file)
        else:
            st.audio(uploaded_file, format=f'audio/{file_extension}')
            content, minutes = process_audio_content(uploaded_file)

    elif file_extension == 'docx':
        content = read_docx(uploaded_file)
        llm = initialize_language_model()
        chain = create_analysis_chain(llm)
        minutes = generate_meeting_minutes(chain, content)

    elif file_extension == 'pdf':
        content = read_pdf(io.BytesIO(uploaded_file.getvalue()))
        llm = initialize_language_model()
        chain = create_analysis_chain(llm)
        minutes = generate_meeting_minutes(chain, content)

    return content, minutes

def main():
    st.title("Meeting Minutes Generator")

    # Check for FFmpeg installation
    if not check_ffmpeg():
        st.error("FFmpeg is not installed. Please contact the administrator.")
        st.stop()

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload File", "Record Audio"])

    with tab1:
        uploaded_file = st.file_uploader("Choose a file", type=['mp3', 'wav', 'm4a', 'docx', 'pdf'])

        if uploaded_file is not None:
            st.write(f"{uploaded_file.name} uploaded")

            if st.button("Generate Minutes from File"):
                with st.spinner("Processing file..."):
                    content, minutes = handle_uploaded_file(uploaded_file)
                display_results(content, minutes)

    with tab2:
        st.write("Record your meeting audio")
        st.write("Click the microphone button below to start recording:")

        # Add audio recorder
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x"
        )

        if audio_bytes:
            st.audio(audio_bytes, format='audio/mp3')

            if st.button("Generate Minutes from Recording"):
                with st.spinner("Processing recording..."):
                    # Save the recorded audio to a temporary file
                    temp_audio_file = save_audio_file(audio_bytes)

                    # Process the audio file
                    with open(temp_audio_file, 'rb') as audio_file:
                        content, minutes = process_audio_content(audio_file)

                    # Remove temporary audio file
                    os.unlink(temp_audio_file)

                display_results(content, minutes)

if __name__ == "__main__":
    main()
