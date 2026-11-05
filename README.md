# YouTube Comment Sentiment Analyzer

## Execution Instructions

### 1. Clone the Repository

```powershell
git clone https://github.com/xxslade/YTCommentSentiment_v2.git
cd YTCommentSentiment_v2
```

### 2. Create and Activate a Virtual Environment

We can create a virtual environment to install and manage the project dependencies separately.

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the Required Packages

We can install all the required Python packages using the provided `requirements.txt` file.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure the YouTube Data API Key

To fetch comments from YouTube videos, we need a YouTube Data API key.

Create a `.env` file in the project root directory and add the API key:

```env
YOUTUBE_API_KEY=your_api_key_here
```

The API key is configured locally and is not included in the GitHub repository.

### 5. Run a Sample Comment Prediction

We can test the trained model using the sample comment provided in `predict.py`.

From the project root directory, run:

```powershell
python src/predict.py
```

The script loads the saved model and displays the predicted labels and their corresponding probabilities.

### 6. Analyze Comments from a YouTube Video

The video analysis module is available in `src/analyzeVideo.py`. It contains the functions for retrieving YouTube comments, classifying them using the trained model, and generating a category-wise summary.

The YouTube Data API key configured in Step 4 is required for retrieving comments.

## Notes

- The trained model and tokenizer are included in `outputs/finalModel/`.
- The project can run on a CPU; a dedicated GPU is not required.
- The current implementation retrieves top-level YouTube comments only.