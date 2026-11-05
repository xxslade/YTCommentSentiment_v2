from youtubeApi import extractVideoId, fetchComments
from predict import CommentPredictor
import matplotlib.pyplot as plt

def analyzeVideo(videoUrl, maxComments=100):
    videoId = extractVideoId(videoUrl)

    if not videoId:
        raise ValueError("Invalid YouTube video URL.")

    comments = fetchComments(videoId, maxComments)

    if not comments:
        print("No comments found.")
        return []

    predictor = CommentPredictor()

    results = []

    for comment in comments:
        predictions = predictor.predict(comment)

        results.append({
            "comment": comment,
            "predictions": predictions
        })

    return results

def summarizeResults(results):
    totalComments = len(results)

    if totalComments == 0:
        return {}

    summary = {}

    for label in results[0]["predictions"]:
        positiveCount = 0
        totalProbability = 0

        for result in results:
            prediction = result["predictions"][label]

            totalProbability += prediction["probability"]

            if prediction["predicted"]:
                positiveCount += 1

        summary[label] = {
            "count": positiveCount,
            "percentage": (positiveCount / totalComments) * 100,
            "averageProbability": totalProbability / totalComments
        }

    return summary

def plotSummary(summary):
    labels = list(summary.keys())
    counts = [stats["count"] for stats in summary.values()]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)

    plt.xlabel("Comment Category")
    plt.ylabel("Number of Comments")
    plt.title("YouTube Comment Sentiment Analysis")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    videoUrl = "https://www.youtube.com/watch?v=EYkbg444H8A"

    results = analyzeVideo(videoUrl, maxComments=20)

    print("\nTotal comments analyzed:", len(results))

    summary = summarizeResults(results)

    print("\n========== VIDEO SENTIMENT SUMMARY ==========")

    for label, stats in summary.items():
        print(f"\n{label}")
        print(f"  Count: {stats['count']}")
        print(f"  Percentage: {stats['percentage']:.2f}%")
        print(f"  Average probability: {stats['averageProbability']:.4f}")

    print("\n==============================================")

    plotSummary(summary)