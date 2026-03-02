import subprocess  #used to run shell commands.
import whisper  
#- Imports OpenAI’s Whisper ASR (Automatic Speech Recognition) library.
# Used to transcribe audio to text.

import os
#- Provides access to file system operations like checking if a file exists or deleting it


# subprocess is used to execute shell commands

#- Defines a function to extract audio from a video file.
#- Takes the video path and optional output audio path.

def extract_audio(video_path: str, audio_path: str = "temp_audio.wav") -> str:
    #- Deletes any existing audio file to avoid overwriting issues
    if os.path.exists(audio_path):
        os.remove(audio_path)

    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        audio_path,
        "-y"
    ]

    '''
- Constructs the ffmpeg command:
- -i: input file
- -q:a 0: best audio quality
- -map a: selects audio stream
- -y: overwrite output file
'''

    #- Executes the command silently.
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
   
    return audio_path


#- Transcribes the audio using Whisper.- model_size can be "tiny", "base", "small", etc.

def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    #Loads the Whisper model.
    model = whisper.load_model(model_size)
    #Transcribes the audio file
    result = model.transcribe(audio_path)
    #- Extracts the transcript text from the result dictionary.
    transcript = result["text"]
    #Returns the transcript.
    return transcript