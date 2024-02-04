from openai import OpenAI

os.environ['OPENAI_API_KEY'] = constants.OPENAI_API_KEY

def convert_audio_to_text(audio_file):
    client = OpenAI()

    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file, 
        response_format="text"
    )

    
