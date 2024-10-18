# Meeting Minutes Generator

This repository contains a Streamlit application that generates comprehensive meeting minutes from audio files (MP3, WAV, M4A), DOCX, or PDF documents. It uses speech recognition for audio files and text extraction for documents, then leverages AI to create detailed meeting minutes.

## Features

- Supports multiple input formats: MP3, WAV, M4A, DOCX, and PDF
- Transcribes audio files using Whisper API
- Generates meeting minutes using GPT-4o
- Provides downloadable output in both DOCX and PDF formats

## Prerequisites

- Python 3.9+
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:
   
   ```
   https://github.com/taham655/minutes-of-meetings-generator.git
   cd meeting-minutes-generator
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Export the required API keys:
   ```
   export GROQ_API_KEY='your_groq_api_key_here'
   export OPENAI_API_KEY='your_openai_api_key_here'
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload your audio file (MP3, WAV, M4A), DOCX, or PDF document.

4. Click on "Generate Meeting Minutes" to process the file and create the minutes.

5. View the generated minutes and download them in DOCX or PDF format.

## Note

Ensure that FFmpeg is installed on your system. If it's not installed, the application will display an error message.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
