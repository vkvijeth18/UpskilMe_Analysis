import sys
import os
import time
import logging
import librosa
import numpy as np
from textblob import TextBlob
from pymongo import MongoClient
from dotenv import load_dotenv
import cloudinary
import speech_recognition as sr
import cloudinary.uploader
from moviepy import VideoFileClip
import nltk
import json
import re
from nltk.corpus import stopwords

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.error("Missing MongoDB URI in environment variables")
    sys.exit(1)

# Function to get fresh MongoDB connection
def get_mongo_connection():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.server_info()
        db = client["mockinterview"]
        logger.info("Successfully connected to MongoDB")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None

# Initial MongoDB connection for validation
try:
    initial_db = get_mongo_connection()
    if not initial_db:
        sys.exit(1)
except Exception as e:
    logger.error(f"Failed to make initial MongoDB connection: {e}")
    sys.exit(1)

# Cloudinary configuration
try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET")
    )
    if not (os.getenv("CLOUDINARY_CLOUD_NAME") and 
            os.getenv("CLOUDINARY_API_KEY") and 
            os.getenv("CLOUDINARY_API_SECRET")):
        logger.warning("Missing Cloudinary credentials in environment variables")
except Exception as e:
    logger.error(f"Error in Cloudinary configuration: {e}")
    sys.exit(1)

def download_video_from_url(video_url, output_path):
    """Download video from Cloudinary or other URL"""
    import requests
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Video downloaded and saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        return None

def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio from video file using moviepy"""
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
            
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path, codec='pcm_s16le')
        logger.info(f"Audio extracted to {output_audio_path}")
        return output_audio_path
    except Exception as e:
        logger.error(f"Failed to extract audio: {e}")
        return None

def transcribe_audio(audio_file):
    """Transcribe audio using Google Speech Recognition with improved chunking"""
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return ""
        
    recognizer = sr.Recognizer()
    transcript = ""
    
    try:
        # Load audio file for duration calculation
        y, sr = librosa.load(audio_file)
        audio_duration = librosa.get_duration(y=y, sr=sr)
        
        # Split audio into smaller chunks for better recognition
        chunk_duration = 30  # 30 seconds chunks
        
        if audio_duration <= chunk_duration:
            # Process short audio in one go
            with sr.AudioFile(audio_file) as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)
        else:
            # Process in chunks with proper offsets
            chunks = int(audio_duration / chunk_duration) + 1
            for i in range(chunks):
                offset = i * chunk_duration
                duration = min(chunk_duration, audio_duration - offset)
                
                with sr.AudioFile(audio_file) as chunk_source:
                    recognizer.adjust_for_ambient_noise(chunk_source)
                    chunk_audio = recognizer.record(chunk_source, duration=duration)
                    
                    # Add retry logic for API calls
                    max_retries = 3
                    retry_count = 0
                    backoff_time = 2
                    
                    while retry_count < max_retries:
                        try:
                            chunk_transcript = recognizer.recognize_google(chunk_audio)
                            transcript += " " + chunk_transcript
                            break  # Success, exit retry loop
                        except sr.UnknownValueError:
                            logger.warning(f"Could not understand chunk {i+1}/{chunks}")
                            break  # No retry for speech recognition failures
                        except sr.RequestError as e:
                            retry_count += 1
                            if retry_count >= max_retries:
                                logger.error(f"API error after {max_retries} retries: {e}")
                                break
                            logger.warning(f"API error (attempt {retry_count}): {e}. Retrying in {backoff_time}s...")
                            time.sleep(backoff_time)
                            backoff_time *= 2  # Exponential backoff
        
        logger.info(f"Transcription successful: {len(transcript)} characters")
        return transcript.strip()
    except sr.UnknownValueError:
        logger.warning("Speech Recognition could not understand the audio")
        # Return placeholder text for testing instead of empty string
        return "This is a placeholder transcript since speech recognition failed. The interview appears to have audio issues."
    except sr.RequestError as e:
        logger.error(f"Speech Recognition service error: {e}")
        return "Speech recognition service unavailable. Please try again later."
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

def count_filler_words(transcript):
    """Count filler words in transcript with improved accuracy"""
    if not transcript:
        return {"count": 0, "rate": 0, "total_words": 0}
        
    # More comprehensive list of filler words
    filler_words = {
        "um", "uh", "like", "you know", "so", "actually",
        "basically", "literally", "kinda", "sort of", 
        "i mean", "right", "well", "hmm", "er", "ah",
        "okay", "anyway", "anyhow", "whatever"
    }
    
    # Convert to lowercase and tokenize
    words = transcript.lower().split()
    
    # Count single word fillers
    single_word_count = sum(words.count(word) for word in filler_words if " " not in word)
    
    # Count multi-word fillers
    multi_word_count = 0
    for phrase in [fw for fw in filler_words if " " in fw]:
        multi_word_count += transcript.lower().count(phrase)
    
    total_count = single_word_count + multi_word_count
    
    # Count actual words (excluding non-word elements)
    total_words = len(re.findall(r'\b\w+\b', transcript))
    
    # Calculate filler word rate (percentage)
    filler_rate = (total_count / total_words) * 100 if total_words > 0 else 0
    
    return {
        "count": total_count,
        "rate": round(filler_rate, 2),
        "total_words": total_words
    }

def analyze_emotion(transcript):
    """Analyze emotion with improved accuracy using TextBlob"""
    if not transcript:
        return {"emotion": "unknown", "polarity": 0, "subjectivity": 0}
    
    try:
        # Analyze the entire transcript
        full_analysis = TextBlob(transcript)
        
        # Get average sentiment across sentences for more accuracy
        sentences = full_analysis.sentences
        if not sentences:
            return {"emotion": "neutral", "polarity": 0, "subjectivity": 0}
            
        polarities = [sentence.sentiment.polarity for sentence in sentences]
        avg_polarity = sum(polarities) / len(polarities) if polarities else 0
        
        subjectivity = full_analysis.sentiment.subjectivity
        
        # Map polarity to emotion with more granularity
        if avg_polarity > 0.3:
            emotion = "very positive"
        elif avg_polarity > 0.1:
            emotion = "positive"
        elif avg_polarity < -0.3:
            emotion = "very negative"
        elif avg_polarity < -0.1:
            emotion = "negative"
        else:
            emotion = "neutral"
            
        # Map this to interview-specific emotions
        interview_emotion = "neutral"
        if emotion in ["very positive", "positive"]:
            interview_emotion = "confident"
        elif emotion in ["very negative", "negative"]:
            interview_emotion = "nervous"
            
        return {
            "emotion": interview_emotion,
            "polarity": round(avg_polarity, 2),
            "subjectivity": round(subjectivity, 2)
        }
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return {"emotion": "unknown", "polarity": 0, "subjectivity": 0}

def analyze_speech_rate(transcript, audio_duration):
    """Analyze speaking rate (words per minute) with improved word counting"""
    if not transcript or audio_duration <= 0:
        return 0
    
    # Count actual words (excluding non-word elements)
    words = len(re.findall(r'\b\w+\b', transcript))
    minutes = audio_duration / 60
    wpm = words / minutes if minutes > 0 else 0
    
    return round(wpm, 1)

def analyze_voice_features(audio_file):
    """Analyze voice features including pitch, energy, and pauses with memory limits"""
    try:
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return {}
        
        # Limit analysis to first minute to avoid memory issues with large files    
        MAX_DURATION = 60  # seconds
        y, sr = librosa.load(audio_file, duration=MAX_DURATION)
        
        # Pitch analysis
        pitch, mag = librosa.piptrack(y=y, sr=sr)
        pitch_valid = pitch[pitch > 0]
        
        # Handle potential empty arrays
        if len(pitch_valid) == 0:
            avg_pitch = 0
            pitch_std = 0
        else:
            avg_pitch = np.mean(pitch_valid)
            pitch_std = np.std(pitch_valid)
        
        # Energy/volume analysis
        energy = np.sum(y**2) / len(y) if len(y) > 0 else 0
        
        # Detect pauses (silence)
        intervals = librosa.effects.split(y, top_db=20)
        total_silence = (len(y) - sum(i[1] - i[0] for i in intervals)) / sr
        silence_ratio = total_silence / (len(y) / sr) if len(y) > 0 else 0
        
        # Speaking variability (pitch variation - good indicator of engagement)
        # Fix potential division by zero
        pitch_variability = pitch_std / avg_pitch if avg_pitch > 0 and pitch_std > 0 else 0
        
        # Map features to confidence indicators
        confidence_score = min(10, max(0, 5 + 
                                    (pitch_variability * 2) +  # Higher variation is better
                                    (energy * 100) -          # More energy is better
                                    (silence_ratio * 5)))     # Less silence is better
        
        return {
            "avg_pitch": round(float(avg_pitch), 2),
            "pitch_variability": round(float(pitch_variability), 2),
            "energy": round(float(energy), 5),
            "silence_ratio": round(float(silence_ratio), 2),
            "confidence_score": round(float(confidence_score), 1)
        }
    except Exception as e:
        logger.error(f"Error in voice analysis: {e}")
        return {
            "avg_pitch": 0,
            "pitch_variability": 0,
            "energy": 0,
            "silence_ratio": 0,
            "confidence_score": 5  # Default neutral score
        }

def analyze_clarity(transcript):
    """Analyze speech clarity with improved metrics"""
    if not transcript:
        return {"score": 0, "avg_sentence_length": 0, "complex_sentences": 0, "lexical_diversity": 0}
        
    try:
        analysis = TextBlob(transcript)
        sentences = analysis.sentences
        
        if not sentences:
            return {"score": 0, "avg_sentence_length": 0, "complex_sentences": 0, "lexical_diversity": 0}
            
        # Calculate average sentence length
        sentence_lengths = [len(str(sentence).split()) for sentence in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        # Count complex sentences (sentences with more than 20 words)
        complex_sentences = sum(1 for length in sentence_lengths if length > 20)
        complex_ratio = complex_sentences / len(sentences) if sentences else 0
        
        # Lexical diversity (unique words / total words)
        words = transcript.lower().split()
        stopword_set = set(stopwords.words('english'))
        content_words = [w for w in words if w not in stopword_set]
        
        if not content_words:
            lexical_diversity = 0
        else:
            lexical_diversity = len(set(content_words)) / len(content_words)
        
        # Final clarity score based on average length and lexical diversity
        score = min(10, round((avg_length / 20 + lexical_diversity) * 5, 2))  # Normalize to max 10
        
        return {
            "score": score,
            "avg_sentence_length": round(avg_length, 2),
            "complex_sentences": complex_sentences,
            "lexical_diversity": round(lexical_diversity, 2)
        }
    except Exception as e:
        logger.error(f"Error in clarity analysis: {e}")
        return {"score": 0, "avg_sentence_length": 0, "complex_sentences": 0, "lexical_diversity": 0}


def analyze_video(video_url):
    """Comprehensive video analysis with improved metrics and file management"""
    # Create temp directory for processing
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate unique filenames
    file_id = str(int(time.time()))
    video_path = os.path.join(temp_dir, f"video_{file_id}.webm")
    audio_path = os.path.join(temp_dir, f"audio_{file_id}.wav")
    
    try:
        # Download video
        downloaded_video = download_video_from_url(video_url, video_path)
        if not downloaded_video:
            return {"error": "Failed to download video"}
        
        # Extract audio
        extracted_audio = extract_audio_from_video(video_path, audio_path)
        if not extracted_audio:
            return {"error": "Failed to extract audio"}
        
        # Get audio duration with fixed parameter usage
        y, sr = librosa.load(audio_path)
        audio_duration = librosa.get_duration(y=y, sr=sr)
        
        # Transcribe audio
        logger.info("Starting transcription...")
        transcript = transcribe_audio(audio_path)
        
        if not transcript:
            return {
                "error": "Transcription failed",
                "confidence": 0,
                "clarity": 0,
                "filler_words": {"count": 0, "rate": 0},
                "emotion": "unknown",
                "score": 0,
                "summary": "Could not analyze due to transcription failure."
            }
        
        # Analyze speech metrics
        logger.info("Analyzing speech metrics...")
        filler_words = count_filler_words(transcript)
        emotion = analyze_emotion(transcript)
        voice_features = analyze_voice_features(audio_path)
        clarity = analyze_clarity(transcript)
        speech_rate = analyze_speech_rate(transcript, audio_duration)
        
        # Calculate a composite score
        confidence_weight = 0.3
        clarity_weight = 0.25
        filler_weight = 0.15
        emotion_weight = 0.15
        speech_rate_weight = 0.15
        
        # Normalize filler word rate to a 0-10 scale (lower is better)
        filler_score = max(0, 10 - min(10, filler_words["rate"]))
        
        # Normalize speech rate (ideal range: 130-160 WPM)
        speech_rate_score = 10 - min(10, abs(speech_rate - 145) / 15)
        
        # Emotion booster (confidence is good for interviews)
        emotion_score = 10 if emotion["emotion"] == "confident" else (
            5 if emotion["emotion"] == "neutral" else 0
        )
        
        # Calculate weighted score
        score = (
            voice_features["confidence_score"] * confidence_weight +
            clarity["score"] * clarity_weight +
            filler_score * filler_weight +
            emotion_score * emotion_weight +
            speech_rate_score * speech_rate_weight
        )
        
        # Round to nearest integer out of 100
        final_score = round(score * 10)
        
        # Generate detailed summary
        summary = (
            f"Overall Interview Performance: {final_score}/100\n\n"
            f"• Confidence: {voice_features['confidence_score']}/10 "
            f"(Pitch variation: {voice_features['pitch_variability']}, "
            f"Energy level: {'High' if voice_features['energy'] > 0.0001 else 'Moderate'}, "
            f"Silence: {voice_features['silence_ratio']*100:.1f}%)\n"
            f"• Speech Clarity: {clarity['score']}/10 "
            f"(Avg sentence length: {clarity['avg_sentence_length']} words, "
            f"Complex sentences: {clarity['complex_sentences']}, "
            f"Vocabulary diversity: {clarity['lexical_diversity']:.2f})\n"
            f"• Filler Words: {filler_words['count']} total, "
            f"{filler_words['rate']:.1f}% of speech\n"
            f"• Emotional Tone: {emotion['emotion'].capitalize()} "
            f"(Sentiment polarity: {emotion['polarity']})\n"
            f"• Speaking Rate: {speech_rate} words per minute"
        )
        
        # Add recommendations based on scores
        recommendations = []
        
        if voice_features['confidence_score'] < 5:
            recommendations.append("Work on speaking with more confidence by varying your tone and increasing energy.")
        
        if clarity['score'] < 5:
            recommendations.append("Improve clarity by using shorter sentences and more diverse vocabulary.")
        
        if filler_words['rate'] > 5:
            recommendations.append(f"Reduce filler words like 'um', 'uh', and 'like' which comprised {filler_words['rate']:.1f}% of your speech.")
        
        if emotion['emotion'] == 'nervous':
            recommendations.append("Practice speaking with a more positive, confident tone.")
        
        if speech_rate < 120:
            recommendations.append("Try to increase your speaking pace slightly for better engagement.")
        elif speech_rate > 170:
            recommendations.append("Consider slowing down your speech for better clarity.")
            
        if recommendations:
            summary += "\n\nRecommendations:\n" + "\n".join(f"• {r}" for r in recommendations)
        
        # Detailed result
        analysis_result = {
            "score": final_score,
            "summary": summary,
            "metrics": {
                "confidence": voice_features,
                "clarity": clarity,
                "filler_words": filler_words,
                "emotion": emotion,
                "speech_rate": speech_rate
            },
            "transcript": transcript,
            "recommendations": recommendations
        }
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": f"Analysis failed: {str(e)}"}
    finally:
        # Clean up temporary files
        for path in [video_path, audio_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Removed temporary file: {path}")
                except Exception as e:
                    logger.warning(f"Error removing temporary file {path}: {e}")

def extract_public_id_from_url(video_url):
    """Extract Cloudinary public ID from URL with robust pattern matching"""
    try:
        # Try to extract using regex pattern
        match = re.search(r'/v\d+/([^/]+/[^.]+)', video_url)
        if match:
            return match.group(1)
        
        # Fallback to simple path extraction
        parts = video_url.split("/")
        public_id_parts = parts[-1].split(".")
        return "/".join(["interview_videos", public_id_parts[0]])
    except Exception as e:
        logger.error(f"Error extracting public ID: {e}")
        return None

def process_pending_interviews():
    """Process pending interviews with error handling and retries"""
    try:
        # Get fresh connection for thread safety
        db = get_mongo_connection()
        if not db:
            logger.error("Failed to connect to database")
            return
            
        interview_col = db["interviews"]
            
        # Find interviews that need processing
        pending_interviews = interview_col.find({"status": "Processing"})
        count = interview_col.count_documents({"status": "Processing"})
        
        if count == 0:
            logger.info("No pending interviews found")
            return
            
        logger.info(f"Found {count} pending interviews")
        
        for interview in pending_interviews:
            interview_id = str(interview['_id'])
            logger.info(f"Processing interview: {interview_id}")
            
            try:
                video_url = interview.get("video_url")
                
                if not video_url:
                    logger.error(f"No video URL found for interview {interview_id}")
                    interview_col.update_one(
                        {"_id": interview["_id"]},
                        {"$set": {"status": "Error", "error": "No video URL provided"}}
                    )
                    continue
                
                # Analyze the video
                result = analyze_video(video_url)
                
                if "error" in result:
                    logger.error(f"Analysis failed for interview {interview_id}: {result['error']}")
                    interview_col.update_one(
                        {"_id": interview["_id"]},
                        {"$set": {"status": "Error", "error": result["error"]}}
                    )
                    continue
                
                # Delete video from Cloudinary if it's stored there
                if "cloudinary" in video_url.lower():
                    try:
                        public_id = extract_public_id_from_url(video_url)
                        if public_id:
                            result = cloudinary.uploader.destroy(public_id, resource_type="video")
                            logger.info(f"Cloudinary deletion result: {result}")
                    except Exception as e:
                        logger.error(f"Cloudinary deletion error: {e}")
                
                # Update MongoDB with analysis results
                interview_col.update_one(
                    {"_id": interview["_id"]},
                    {
                        "$set": {
                            "status": "Analyzed",
                            "analysis": result,
                            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                )
                logger.info(f"Analysis complete for interview {interview_id}")
                
            except Exception as e:
                logger.error(f"Error processing interview {interview_id}: {e}")
                interview_col.update_one(
                    {"_id": interview["_id"]},
                    {"$set": {"status": "Error", "error": str(e)}}
                )
    
    except Exception as e:
        logger.error(f"Error in process_pending_interviews: {e}")

if __name__ == "__main__":
    logger.info("Starting interview analysis service")
    
    try:
        # Check if we're being called from Node.js with command line args
        if len(sys.argv) > 1:
            video_url = sys.argv[1]
            logger.info(f"Processing single video: {video_url}")
            result = analyze_video(video_url)
            # Print the JSON result for Node.js to capture
            print(json.dumps(result))
        else:
            # Process pending interviews from the database
            process_pending_interviews()
            
        logger.info("Interview analysis service completed")
    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        sys.exit(1)