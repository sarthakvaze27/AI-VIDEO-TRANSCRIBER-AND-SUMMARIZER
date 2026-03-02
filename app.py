import streamlit as st
import os
import tempfile
from main import video_to_summary

def main():
    st.title("Video Summarizer AI")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Save to a temporary directory
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.write("Transcribing and Summarizing... This may take a few moments.")

        # Run the pipeline
        summary_result = video_to_summary(
            video_path=video_path,
            model_size="base",
            summarizer_model_name="facebook/bart-large-cnn",
            use_chunking=True
        )

        st.subheader("Summary:")
        st.write(summary_result)

        # Clean up the temporary file
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    main()
