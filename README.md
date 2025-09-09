# E-consultation Sentiment Analysis MVP 

A powerful Streamlit web application for analyzing stakeholder comments and feedback from e-consultation processes using Natural Language Processing (NLP) and sentiment analysis.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Features

- Sentiment Analysis: Automatically classify comments as positive, negative, or neutral using TextBlob
- Text Summarization: Generate concise summaries of stakeholder feedback
- Word Cloud Visualization: Visual representation of frequently used terms
- Statistical Insights: Detailed analytics and charts using Plotly
- Provision-wise Analysis: Sentiment breakdown by document sections
- Stakeholder Group Analysis: Analysis by stakeholder categories
- Export Results: Download analysis results in CSV format
- Interactive Dashboard: Modern, user-friendly Streamlit interface

## Live Demo

Try the app online: https://appapppy-phf7nqyfrl66cotfapobfb.streamlit.app/

## 📸 Screenshots

<img width="1920" height="833" alt="Screenshot (585)" src="https://github.com/user-attachments/assets/41f6c051-9d30-4a59-8d75-7cc3aecf0395" />



## 🛠️ Technology Stack

- Frontend: Streamlit
- Backend: Python 3.8+
- NLP: TextBlob, NLTK
- Data Processing: Pandas, NumPy
- Visualization: Plotly, Matplotlib, WordCloud
- Deployment: Streamlit Cloud / Render / Railway

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/aditikamble123/econsultation-sentiment-mvp.git
   cd econsultation-sentiment-mvp
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # Activate on Windows:
   venv\Scripts\activate
   
   # Activate on Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data:
   ```bash
   python -m textblob.download_corpora
   ```

5. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## Usage

### Data Format

Prepare your CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `comment_id` | Unique identifier | 1, 2, 3... |
| `stakeholder_name` | Name of stakeholder | "John Doe" |
| `comment_text` | The actual comment | "This proposal is excellent..." |
| `provision_reference` | Section being commented on | "Section 2.1" |

### Using the Application

1. Upload Data: Click "Browse files" to upload your CSV
2. View Analysis: The app automatically processes and displays:
   - Overall sentiment distribution
   - Detailed sentiment scores
   - Word clouds
   - Statistical summaries
3. Export Results: Download processed data as CSV

## 🚢 Deployment

### Deploy to Streamlit Cloud (Recommended)

1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your forked repository

### Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Click the button above
2. Follow the deployment wizard
3. Your app will be live in minutes

### Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

### Deploy to Heroku

Create a `setup.sh` file:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

Update `Procfile`:
```
web: sh setup.sh && streamlit run streamlit_app.py
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (not tracked in git):
```env
# Optional configurations
MAX_UPLOAD_SIZE=200
STREAMLIT_THEME=light
```

### Streamlit Configuration

Edit `.streamlit/config.toml` for customization:
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
```

## Project Structure

```
econsultation-sentiment-mvp/
├── streamlit_app.py          # Main Streamlit application
├── sentiment_analyzer.py     # Sentiment analysis module
├── text_summarizer.py       # Text summarization module
├── wordcloud_generator.py   # Word cloud generation
├── requirements.txt         # Python dependencies
├── .streamlit/             # Streamlit configuration
│   └── config.toml
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TextBlob for sentiment analysis
- Streamlit for the amazing framework
- NLTK for natural language processing
- All contributors and users

## Contact

Aditi Kamble - [GitHub Profile](https://github.com/aditikamble123)

Project Link: [https://github.com/aditikamble123/econsultation-sentiment-mvp](https://github.com/aditikamble123/econsultation-sentiment-mvp)

## Troubleshooting

### Common Issues

1. ModuleNotFoundError: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. NLTK Data Error: Download required data:
   ```bash
   python -m textblob.download_corpora
   ```

3. Memory Issues: For large datasets, consider:
   - Processing in batches
   - Increasing system memory
   - Using cloud deployment

## Future Enhancements

- [ ] Multi-language support
- [ ] Advanced ML models (BERT, GPT)
- [ ] Real-time analysis
- [ ] API endpoints
- [ ] User authentication
- [ ] PDF report generation
- [ ] Database integration
- [ ] Custom model training

---

⭐ Star this repository if you find it helpful!
