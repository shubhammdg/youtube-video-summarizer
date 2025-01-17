import os
from typing import Optional, Dict
from functools import lru_cache
from urllib.parse import urlparse, parse_qs
import ollama
from youtube_transcript_api import YouTubeTranscriptApi

class YtSummarizer:
    """A class to fetch and summarize YouTube video transcripts."""
    
    def __init__(self, url: str,
                 model: str = 'llama3.2'):
        """
        Initialize the YtSummarizer.
        
        Args:
            url: YouTube video URL\
            model: Language model to use
        """
        if not url:
            raise ValueError("URL cannot be empty")
        self.url = url
        self.model = model

    def get_video_id(self) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        try:
            parsed_url = urlparse(self.url)
            if 'youtube.com' not in parsed_url.netloc:
                raise ValueError("Not a valid YouTube URL")
            
            if 'v' in parse_qs(parsed_url.query):
                return parse_qs(parsed_url.query)['v'][0]
            elif 'youtu.be' in parsed_url.netloc:
                return parsed_url.path[1:]
            else:
                raise ValueError("Could not extract video ID")
        except Exception as e:
            print(f"Error extracting video ID: {str(e)}")
            return None

    @lru_cache(maxsize=100)
    def get_video_transcript(self) -> Optional[str]:
        """Fetch and cache video transcript."""
        video_id = self.get_video_id()
        if not video_id:
            return None
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(segment['text'] for segment in transcript_list)
        except Exception as e:
            print(f"Error getting transcript: {str(e)}")
            return None

    def summarize(self) -> Optional[str]:
        """Summarize the video transcript using OpenAI API."""
        transcript = self.get_video_transcript()
        if not transcript:
            return None

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that summarizes YouTube videos from a transcript. "
                                 "Mention the title of the video and the name of the speaker. "
                                 "Also use bullet points wherever possible to make the summary more readable."
                    }, 
                    {"role": "user", "content": f"Here is the transcript:\n {transcript}"}
                ]
            )
            return response.message.content
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return None


if __name__ == "__main__":
    try:
        yt_summarizer = YtSummarizer("https://www.youtube.com/watch?v=POhK-IlHobc")
        summary = yt_summarizer.summarize()
        if summary:
            print(summary)
        else:
            print("Failed to generate summary")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
