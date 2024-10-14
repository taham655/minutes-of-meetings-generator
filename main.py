import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


filename = os.path.dirname(__file__) + "/output.mp3" # Replace with your audio file!

transcripted = ''
with open(filename, "rb") as file:
    # Create a transcription of the audio file
    transcription = client.audio.transcriptions.create(
      file=(filename, file.read()), 
      model="distil-whisper-large-v3-en"
    )
    
    transcripted = transcription.text
    print(transcripted)


    

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Initialize the language model
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),temperature=0.7)

# Define the initial questions prompt
questions_prompt = ChatPromptTemplate.from_messages([
    ("human", "Please provide the following information about the meeting:\n"
              "1. Where did the meeting take place?\n"
              "2. What is the date of the meeting?")
])

# Define the main analysis prompt
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant tasked with generating comprehensive meeting summaries."),
    ("human", "Based on the following meeting transcript and information, please generate a comprehensive meeting summary:\n"
              "Transcript: {transcript}\n\n"
              "Please provide the following information:\n"
              "1. Minutes of the meeting\n"
              "2. What was discussed\n"
              "3. A brief summary\n"
              "4. The meeting agenda (inferred from the discussion)\n"
              "5. Action items (things to be done)")
])


chain = (
    {"transcript": lambda _: RunnablePassthrough()}
    | analysis_prompt
    | llm
    | StrOutputParser()
)

# Define the main function
def generate_meeting_minutes(transcript):
    return chain.invoke(transcript)

# Example usage
if __name__ == "__main__":
    transcript = transcripted
    minutes = generate_meeting_minutes(transcript)
    print(minutes)
    