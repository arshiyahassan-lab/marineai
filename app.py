import os
import uuid
from flask import Flask, jsonify, request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from openai import OpenAI
import yt_dlp
from dotenv import load_dotenv
from flask_cors import CORS

# -------------------- Load Environment -------------------- #
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Debug: Check if API keys are loaded
print(f"OpenAI API Key loaded: {bool(OPENAI_API_KEY)}")
print(f"YouTube API Key loaded: {bool(YOUTUBE_API_KEY)}")
if OPENAI_API_KEY:
    print(f"OpenAI Key starts with: {OPENAI_API_KEY[:15]}...")
if YOUTUBE_API_KEY:
    print(f"YouTube Key starts with: {YOUTUBE_API_KEY[:15]}...")

# Check for missing API keys
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not found in environment!")
if not YOUTUBE_API_KEY:
    print("ERROR: YOUTUBE_API_KEY not found in environment!")

# -------------------- Initialize Clients -------------------- #
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------- Constants -------------------- #
TOP_PLAYERS = ["Maersk", "CMA CGM", "Hapag-Lloyd", "MSC", "ONE"]

# -------------------- Helper Functions -------------------- #

def get_last_month_youtube_podcasts(search_keywords, max_results=10):
    """Fetch YouTube videos from the last 30 days based on dynamic keywords"""
    if not YOUTUBE_API_KEY:
        print("ERROR: YouTube API key not available")
        return []
    
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        published_after = thirty_days_ago.isoformat("T") + "Z"

        print(f"Searching YouTube for: {search_keywords}")
        
        request = youtube.search().list(
            q=search_keywords,
            part="snippet",
            type="video",
            order="date",
            publishedAfter=published_after,
            maxResults=max_results
        )
        response = request.execute()
        podcasts = []

        for item in response.get("items", []):
            video_id = item["id"].get("videoId")
            if not video_id:
                continue
            podcasts.append({
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "video_url": f"https://www.youtube.com/watch?v={video_id}"
            })

        print(f"Found {len(podcasts)} YouTube videos")
        return podcasts

    except Exception as e:
        print(f"Error fetching YouTube videos: {e}")
        return []

def download_audio(video_url):
    """Download YouTube video audio using yt-dlp"""
    unique_id = str(uuid.uuid4())[:8]
    output_template = f'audio_{unique_id}.%(ext)s'
    
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'extract_flat': False
    }
    
    try:
        print(f"Downloading audio from: {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info)
            
            # Check if the file exists
            if os.path.exists(filename):
                print(f"Downloaded: {filename}")
                return filename
            else:
                # Look for any file with our unique_id
                for file in os.listdir('.'):
                    if unique_id in file and (file.endswith('.m4a') or file.endswith('.mp3') or file.endswith('.webm')):
                        print(f"Found downloaded file: {file}")
                        return file
                raise Exception("Downloaded file not found")
                
    except Exception as e:
        print(f"Error downloading audio from {video_url}: {e}")
        raise

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI Whisper"""
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key not available")
        
    if not os.path.exists(file_path):
        raise Exception(f"Audio file not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    if file_size < 1000:  # Less than 1KB
        raise Exception("Audio file too small - likely corrupted")
    if file_size > 25 * 1024 * 1024:  # 25MB
        raise Exception(f"File too large: {file_size/1024/1024:.1f}MB (max 25MB)")
    
    try:
        print(f"Transcribing: {file_path} (Size: {file_size/1024/1024:.1f}MB)")
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        # Handle different response formats
        transcript_text = transcript if isinstance(transcript, str) else transcript.text if hasattr(transcript, 'text') else str(transcript)
        
        print(f"Transcription successful: {len(transcript_text)} characters")
        return transcript_text
        
    except Exception as e:
        print(f"Transcription failed: {str(e)}")
        raise e
    finally:
        # Clean up the audio file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up: {file_path}")
        except Exception as cleanup_error:
            print(f"Could not clean up {file_path}: {cleanup_error}")

def summarize_text(text):
    """Summarize podcast text using GPT"""
    if not OPENAI_API_KEY:
        return "OpenAI API key not available for summarization"
        
    if not text or len(text.strip()) < 50:
        return "Transcript too short to generate meaningful summary"
    
    # Truncate text if too long to avoid token limits
    max_chars = 12000  # Roughly 3000 tokens
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    prompt = f"""
    Summarize the following podcast transcript, focusing on key shipping industry insights:
    
    Key areas to highlight:
    - Global policy impacts and regulatory changes
    - Political issues affecting shipping
    - Technology advancements and innovations
    - Market trends and business developments
    - Mentions of major players: {', '.join(TOP_PLAYERS)}
    
    Provide a concise summary with actionable insights in bullet points.
    
    Transcript:
    {text}
    """
    
    try:
        print("Generating summary with GPT...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3
        )
        
        summary_text = response.choices[0].message.content
        
        if not summary_text or summary_text.strip() == "":
            return "Summary could not be generated - empty response"
        
        print("Summary generated successfully")
        return summary_text
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Summary generation failed: {str(e)}"

# -------------------- Flask Route (POST) -------------------- #

@app.route("/daily_digest", methods=["POST"])
@app.route("/daily_digest", methods=["POST"])
def daily_digest():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        category = data.get("category", "shipping industry")
        companies_list = data.get("company", TOP_PLAYERS)
        
        # Handle both 'company' and 'companies' keys, and both string and array formats
        if not companies_list:
            companies_list = data.get("companies", TOP_PLAYERS)
        
        # Convert string to array if needed
        if isinstance(companies_list, str):
            companies_list = [companies_list]
        
        if not isinstance(companies_list, list) or not companies_list:
            return jsonify({"error": "company/companies must be a non-empty list or string"}), 400

        # Build search keywords
        search_keywords = f"{category} podcast OR interview OR news " + " OR ".join([f'"{company}"' for company in companies_list])
        print(f"Search query: {search_keywords}")

        # Get YouTube videos
        podcasts = get_last_month_youtube_podcasts(search_keywords=search_keywords, max_results=5)
        
        if not podcasts:
            return jsonify([])  # Return empty array instead of error object

        summaries = []
        processed_count = 0
        max_videos_to_process = 3  # Limit processing to avoid timeouts

        for i, podcast in enumerate(podcasts):
            if processed_count >= max_videos_to_process:
                print(f"Reached maximum processing limit ({max_videos_to_process})")
                break
                
            print(f"\nProcessing video {i+1}/{len(podcasts)}: {podcast['title']}")
            
            try:
                # Download and transcribe
                audio_path = download_audio(podcast["video_url"])
                transcription = transcribe_audio(audio_path)
                
                print(f"Transcription preview: {transcription[:200]}...")
                
                # Generate summary
                summary = summarize_text(transcription)

                summaries.append({
                    "title": podcast["title"],
                    "channel": podcast["channel"],
                    "published_at": podcast["published_at"],
                    "video_url": podcast["video_url"],
                    "summary": summary,
                    "transcript_length": len(transcription)
                })
                
                processed_count += 1
                print(f"✓ Successfully processed: {podcast['title']}")

            except Exception as e:
                error_msg = str(e)
                print(f"✗ Error processing {podcast['title']}: {error_msg}")
                
                summaries.append({
                    "title": podcast.get("title", "Unknown"),
                    "channel": podcast.get("channel", "Unknown"),
                    "published_at": podcast.get("published_at", ""),
                    "video_url": podcast.get("video_url", ""),
                    "summary": f"Processing failed: {error_msg}",
                    "error": error_msg
                })

        print(f"\nCompleted processing {processed_count} videos successfully")
        print(f"Final response: returning array with {len(summaries)} items")
        
        # Return the summaries array directly (not wrapped in an object)
        return jsonify(summaries)

    except Exception as e:
        print(f"Fatal error in daily_digest: {e}")
        return jsonify([])  # Return empty array instead of error objectpyth
# -------------------- Health Check Route -------------------- #

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "openai_key_available": bool(OPENAI_API_KEY),
        "youtube_key_available": bool(YOUTUBE_API_KEY)
    })

# -------------------- Run Flask -------------------- #
if __name__ == "__main__":
    print("Starting Flask application...")
    print(f"OpenAI API configured: {bool(OPENAI_API_KEY)}")
    print(f"YouTube API configured: {bool(YOUTUBE_API_KEY)}")
    app.run(debug=True)
        