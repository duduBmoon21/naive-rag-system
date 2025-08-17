import os
import re
from typing import List
from langchain_core.documents import Document
import yt_dlp

def _parse_transcript(raw_text: str) -> str:
    """
    Clean and normalize raw YouTube transcript text:
    """
    # Remove XML/HTML tags
    text = re.sub(r'<[^>]+>', '', raw_text)
    # Remove bracketed descriptions [CHEERS]
    text = re.sub(r'\[.*?\]', '', text)
    # Remove metadata headers like "Kind: captions Language: en"
    text = re.sub(r'^Kind:.*?\n', '', text, flags=re.MULTILINE)
    # Remove WEBVTT header if present
    text = re.sub(r'^WEBVTT.*\n', '', text, flags=re.MULTILINE)
    # Remove timestamp lines
    text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*\n', '', text)
    
    # Split lines, remove empty lines and duplicates
    lines = text.strip().split('\n')
    cleaned_lines = []
    for line in lines:
        clean_line = line.strip().lstrip('> ').strip()
        if clean_line and (not cleaned_lines or cleaned_lines[-1] != clean_line):
            cleaned_lines.append(clean_line)
    
    # Join and normalize whitespace
    final_text = " ".join(cleaned_lines)
    return re.sub(r'\s+', ' ', final_text).strip()


def load_youtube_transcript(youtube_url: str) -> List[Document]:
    """
    Downloads a transcript from YouTube using yt-dlp (manual or auto captions),
    cleans it, and returns it as a single LangChain Document.
    """
    if not re.match(r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+', youtube_url):
        raise ValueError("Invalid YouTube URL")

    # Options for yt-dlp
    ydl_opts = {
        'format': 'best',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'skip_download': True,
        'quiet': True,
        'ignoreerrors': True,
        'outtmpl': 'temp_transcript_%(id)s',
    }

    transcript_text = ""
    downloaded_file = None
    video_info = {}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_id = info_dict.get('id', 'default')
            video_info = info_dict
            expected_file = f"temp_transcript_{video_id}.en.vtt"

            # Trigger subtitle download
            ydl.download([youtube_url])

            if os.path.exists(expected_file):
                downloaded_file = expected_file
                with open(downloaded_file, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                transcript_text = _parse_transcript(raw_content)
            else:
                raise FileNotFoundError("No VTT transcript downloaded. Ensure captions are available.")

    except Exception as e:
        raise RuntimeError(f"Failed to process YouTube video {youtube_url}: {e}")
    finally:
        if downloaded_file and os.path.exists(downloaded_file):
            os.remove(downloaded_file)

    if not transcript_text:
        raise ValueError("Could not extract any text from the transcript.")

    # Build LangChain Document with metadata
    doc = Document(
        page_content=transcript_text,
        metadata={
            "source": youtube_url,
            "title": video_info.get('title', 'Untitled Video'),
            "duration": video_info.get('duration', 0),
            "views": video_info.get('view_count', 0),
            "type": "youtube",
        }
    )

    return [doc]
