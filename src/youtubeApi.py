import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from urllib.parse import urlparse, parse_qs

load_dotenv()

def createYouTubeClient():
    apiKey = os.getenv("YOUTUBE_API_KEY")

    if not apiKey:
        raise ValueError("YouTube API key not found.")

    return build("youtube", "v3", developerKey=apiKey)

def extractVideoId(videoUrl):
    parsedUrl = urlparse(videoUrl)

    if parsedUrl.hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        if parsedUrl.path == "/watch":
            return parse_qs(parsedUrl.query).get("v", [None])[0]

        if parsedUrl.path.startswith("/shorts/"):
            return parsedUrl.path.split("/")[2]

    elif parsedUrl.hostname == "youtu.be":
        return parsedUrl.path.lstrip("/").split("/")[0]

    return None

def fetchComments(videoId, maxComments=500):
    youtube = createYouTubeClient()

    comments = []
    nextPageToken = None

    try:
        while len(comments) < maxComments:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=videoId,
                maxResults=min(100, maxComments - len(comments)),
                pageToken=nextPageToken,
                textFormat="plainText"
            )

            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            nextPageToken = response.get("nextPageToken")

            if not nextPageToken:
                break

    except HttpError as error:
        print(f"YouTube API error: {error}")
        return []

    return comments

if __name__ == "__main__":
    videoUrl = "https://www.youtube.com/watch?v=-7nfEMjaM1M&list=RDEFLJiV7yOBU&index=16"
    
    videoId = extractVideoId(videoUrl)

    comments = fetchComments(videoId, maxComments=20)

    print("Number of comments:", len(comments))

    for index, comment in enumerate(comments[:5], start=1):
        print(f"{index}. {comment}")

