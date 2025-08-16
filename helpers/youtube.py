import os
import re
import yt_dlp
from typing import List, Optional
from langchain_core.documents import Document

def _clean_transcript(text: str) -> str:
    """Cleans raw transcript text with multiple normalization steps"""
    # Remove various artifacts
    patterns = [
        (r'\d{1,2}:\d{2}(?:\.\d+)?', ''),         # Timestamps (00:00 or 00:00.000)
        (r'\[.*?\]', ''),                          # Sound descriptions [music]
        (r'<.*?>', ''),                            # HTML tags
        (r'^WEBVTT.*$', '', re.MULTILINE),         # WEBVTT header
        (r'^\s*$\n', '', re.MULTILINE),            # Empty lines
        (r'[^\w\s\'",.?!-]', ' '),                # Special chars (keep basic punctuation)
        (r'\s+', ' ')                              # Multiple spaces
    ]
    
    for pattern, repl in patterns:
        if len(pattern) > 2 and pattern.startswith('^') and pattern.endswith('$'):
            text = re.sub(pattern, repl, text, flags=re.MULTILINE)
        else:
            text = re.sub(pattern, repl, text)
    
    return text.strip()

def _download_transcript(url: str, ydl_opts: dict) -> Optional[str]:
    """Downloads and returns cleaned transcript text"""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return None
                
            temp_file = f"temp_{info['id']}.vtt"
            ydl.download([url])
            
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    content = _clean_transcript(f.read())
                os.remove(temp_file)
                return content if content else None
    except Exception:
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return None

def load_youtube_transcript(url: str) -> List[Document]:
    """Loads YouTube transcript with multiple fallback strategies"""
    if not re.match(r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+', url):
        raise ValueError("Invalid YouTube URL. Please use full YouTube links.")
    
    # Strategy 1: Try manual English captions first
    content = _download_transcript(url, {
        'skip_download': True,
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'quiet': True
    })
    
    # Strategy 2: Try automatic English captions
    if not content:
        content = _download_transcript(url, {
            'skip_download': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'quiet': True
        })
    
    if not content:
        raise ValueError(
            "No English captions available. Try a different video with "
            "captions enabled or check if the video is age-restricted."
        )
    
    # Get video metadata
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False) or {}
    
    return [Document(
        page_content=content,
        metadata={
            "source": url,
            "title": info.get('title', 'Untitled Video'),
            "duration": info.get('duration', 0),
            "views": info.get('view_count', 0),
            "type": "youtube",
            "captions_type": "manual" if 'writesubtitles' in ydl_opts else "auto"
        }
    )]