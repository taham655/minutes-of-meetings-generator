import streamlit as st
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
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

if not check_ffmpeg():
    st.error("FFmpeg is not installed. Please contact the administrator.")
    st.stop()

def initialize_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def transcribe_audio(client, file):
    transcription = client.audio.transcriptions.create(
        file=file,
        model="whisper-large-v3",
        language="en",
        temperature=0.2
    )
    return transcription.text

def initialize_language_model():
    return ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model='gpt-4o', temperature=0.3)

def create_analysis_chain(llm):
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant specialized in generating detailed and well-structured meeting minutes. 
Your task is to analyze meeting transcripts and produce comprehensive, professional-grade meeting documentation.

Important Guidelines:
- Be thorough yet concise in your analysis
- Maintain professional language and formatting
- Highlight key decisions and action items
- Identify and categorize discussion topics
- Capture deadlines and assignments clearly
- Note any follow-up meetings or dependencies
- Preserve important technical details and numbers
- Include all participants' significant contributions"""),
        
        ("human", """Based on the following meeting transcript, please generate a comprehensive meeting report with the following structure:

Meeting Metadata:
----------------
- Date: [Extract or indicate if not mentioned]
- Time: [Start and end time if available]
- Location: [Physical location or virtual platform]
- Meeting Type: [Regular/Special/Emergency/Follow-up]
- Attendees: [List with roles if mentioned]
- Facilitator/Chair: [If mentioned]
- Note Taker: [If mentioned]

Transcript: {transcript}

Please provide a detailed analysis with the following sections:

1. Executive Summary (50-100 words)
   - Key objectives achieved
   - Major decisions made
   - Critical announcements

2. Meeting Agenda
   - Reconstruct the agenda based on discussion flow
   - Note if items were added during the meeting
   - Identify tabled items

3. Detailed Discussion Points
   - Break down by topic/theme
   - Include:
     * Main arguments and perspectives
     * Data or statistics presented
     * Questions raised and answers provided
     * Concerns or challenges discussed
     * Solutions proposed
     * Decisions reached
     * Voting results (if any)

4. Action Items
   - List all tasks with:
     * Task description
     * Assignee(s)
     * Priority level
     * Due date
     * Dependencies
     * Success criteria
     * Status (if mentioned)

5. Resource Allocation
   - Budget discussions
   - Resource assignments
   - Tool or system requirements

6. Risk and Issues
   - Identified risks
   - Mitigation strategies
   - Escalated issues
   - Blockers

7. Next Steps
   - Immediate actions required
   - Follow-up meeting details
   - Dependencies on other teams/projects

8. Additional Notes
   - Parking lot items
   - References to documents/presentations
   - Links to relevant resources
   - Related meetings

Format Requirements:
-------------------
- Use clear headings and subheadings
- Bullet points for better readability
- Bold text for emphasis on key points
- Include page numbers for longer reports
- Use tables where appropriate
- Maintain consistent formatting

Special Instructions:
--------------------
1. For Technical Meetings:
   - Preserve technical terminology
   - Include system names and versions
   - Note architectural decisions
   - Document API changes or database updates

2. For Strategy Meetings:
   - Highlight market analysis
   - Include competitor information
   - Note strategic initiatives
   - Capture timeline milestones

3. For Project Updates:
   - Include project metrics
   - Note timeline changes
   - List blocked items
   - Document scope changes

4. For Decision-Making Meetings:
   - Detail options considered
   - Document voting results
   - Include dissenting opinions
   - Note approval chain

Length Guidelines:
-----------------
- Short Meetings (<30 mins): 1-2 pages
- Medium Meetings (30-60 mins): 2-4 pages
- Long Meetings (>60 mins): 4+ pages with section breaks

Confidentiality Notice:
----------------------
- Mark sensitive information
- Note distribution restrictions
- Include classification level if applicable

Quality Checks:
--------------
- Verify all action items have owners
- Ensure dates are clearly specified
- Confirm all decisions are documented
- Check for clarity and completeness
- Validate technical accuracy
- Review for consistent terminology""")
    ])
    
    chain = (
        {"transcript": lambda _: RunnablePassthrough()}
        | analysis_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def generate_meeting_minutes(chain, content):
    return chain.invoke(content)

def read_docx(file):
    return docx2txt.process(file)

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def save_as_docx(content):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
        doc = Document()
        doc.add_paragraph(content)
        doc.save(tmp.name)
        return tmp.name

def save_as_pdf(content):
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
    with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_m4a:
        temp_m4a.write(input_file.getvalue())
        temp_m4a_path = temp_m4a.name

    output_mp3 = temp_m4a_path.replace('.m4a', '.mp3')
    
    audio = AudioSegment.from_file(temp_m4a_path, format="m4a")
    audio.export(output_mp3, format="mp3")
    
    os.unlink(temp_m4a_path)  # Remove the temporary M4A file
    
    return output_mp3

def main():
    st.title("Meeting Minutes Generator")

    uploaded_file = st.file_uploader("Choose a file", type=['mp3', 'wav', 'm4a', 'docx', 'pdf'])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension in ['mp3', 'wav', 'm4a']:
            if file_extension == 'm4a':
                with st.spinner("Converting M4A to MP3..."):
                    mp3_file = convert_m4a_to_mp3(uploaded_file)
                st.audio(mp3_file, format='audio/mp3')
                content_type = "audio"
            else:
                st.audio(uploaded_file, format=f'audio/{file_extension}')
                content_type = "audio"
        elif file_extension == 'docx':
            st.write("DOCX file uploaded")
            content_type = "docx"
        elif file_extension == 'pdf':
            st.write("PDF file uploaded")
            content_type = "pdf"

        if st.button("Generate Meeting Minutes"):
            with st.spinner("Processing file..."):
                if content_type == "audio":
                    groq_client = initialize_groq_client()
                    if file_extension == 'm4a':
                        with open(mp3_file, 'rb') as audio_file:
                            content = transcribe_audio(groq_client, audio_file)
                        os.unlink(mp3_file)  # Remove the temporary MP3 file
                    else:
                        content = transcribe_audio(groq_client, uploaded_file)
                elif content_type == "docx":
                    content = read_docx(uploaded_file)
                elif content_type == "pdf":
                    content = read_pdf(io.BytesIO(uploaded_file.getvalue()))

            st.subheader("Content")
            st.text_area("Full Content", content, height=200)

            with st.spinner("Generating meeting minutes..."):
                llm = initialize_language_model()
                chain = create_analysis_chain(llm)
                minutes = generate_meeting_minutes(chain, content)

            st.subheader("Meeting Minutes")
            st.markdown(minutes)

            # Generate DOCX and PDF files
            docx_file = save_as_docx(minutes)
            pdf_file = save_as_pdf(minutes)

            col1, col2 = st.columns(2)

            # Download buttons for DOCX and PDF
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

if __name__ == "__main__":
    main()
