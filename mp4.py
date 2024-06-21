# from moviepy.editor import VideoFileClip
#
#
# def extract_audio_from_video(video_file_path, output_audio_file_path):
#     """
#     Extracts audio from an MP4 video file and saves it as a WAV file, preserving the audio quality without any loss.
#
#     Args:
#     - video_file_path: Path to the input video file (e.g., 'input_video.mp4').
#     - output_audio_file_path: Path where the extracted audio should be saved (e.g., 'output_audio.wav').
#     """
#     # Load the video file
#     video_clip = VideoFileClip(video_file_path)
#
#     # Extract the audio part
#     audio_clip = video_clip.audio
#
#     # Save the audio in WAV format
#     audio_clip.write_audiofile(output_audio_file_path, codec='pcm_s16le')
#
#     # Close the clips to free up system resources
#     audio_clip.close()
#     video_clip.close()
#
#
# # Example usage
# video_file_path = 'soul-power.mp4'
# output_audio_file_path = 'soul-power.wav'
# extract_audio_from_video(video_file_path, output_audio_file_path)
from moviepy.editor import VideoFileClip


def extract_audio_from_video(video_file_path, output_audio_file_path):
    """
    Extracts audio from an MP4 video file and saves it as an MP3 file.

    Args:
    - video_file_path: Path to the input video file (e.g., 'input_video.mp4').
    - output_audio_file_path: Path where the extracted audio should be saved (e.g., 'output_audio.mp3').
    """
    # Load the video file
    video_clip = VideoFileClip(video_file_path)

    # Extract the audio part
    audio_clip = video_clip.audio

    # Save the audio
    audio_clip.write_audiofile(output_audio_file_path, codec='libmp3lame')

    # Close the clips to free up system resources
    audio_clip.close()
    video_clip.close()


# Example usage
video_file_path = 'soul-power.mp4'
output_audio_file_path = 'soul-power.mp3'
extract_audio_from_video(video_file_path, output_audio_file_path)
