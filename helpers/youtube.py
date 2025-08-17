import re
from typing import List, Optional
from langchain_core.documents import Document
import yt_dlp

def _clean_transcript(text: str) -> str:
    """Clean transcript text, removing timestamps, HTML, empty lines, etc."""
    import re
    patterns = [
        (r'\d{1,2}:\d{2}(?:\.\d+)?', ''),   # timestamps
        (r'\[.*?\]', ''),                    # sound descriptions
        (r'<.*?>', ''),                      # HTML tags
        (r'^WEBVTT.*$', '', re.MULTILINE),   # WEBVTT header
        (r'^\s*$\n', '', re.MULTILINE),     # empty lines
        (r'[^\w\s\'",.?!-]', ' '),           # special chars
        (r'\s+', ' ')                        # multiple spaces
    ]
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)
    return text.strip()

def _download_transcript(url: str, ydl_opts: dict) -> Optional[str]:
    """Download and clean transcript using yt_dlp"""
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
        return None

def load_youtube_transcript(url: str) -> List[Document]:
    """Load YouTube transcript with multiple fallbacks"""
    if not re.match(r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+', url):
        raise ValueError("Invalid YouTube URL")

    # Try manual English captions
    content = _download_transcript(url, {
        'skip_download': True,
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'quiet': True
    })

    # Fallback to automatic captions
    if not content:
        content = _download_transcript(url, {
            'skip_download': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'quiet': True
        })

    if not content:
        raise ValueError(
            "No English captions available. Use a video with captions enabled."
        )

    # Video metadata
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
            "captions_type": "manual" if 'writesubtitles' in locals() else "auto"
        }
    )]
