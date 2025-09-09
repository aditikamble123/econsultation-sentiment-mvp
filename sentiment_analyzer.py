"""
Sentiment Analysis Module for E-consultation Comments
"""

import pandas as pd
from textblob import TextBlob
import nltk
from collections import Counter
import re

# Download required NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class SentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(['the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'in', 
                               'to', 'for', 'of', 'with', 'as', 'by', 'that', 'this', 
                               'it', 'from', 'or', 'but', 'are', 'was', 'were', 'be',
                               'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                               'could', 'should', 'may', 'might', 'must', 'can', 'shall'])
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a single text using TextBlob
        Returns sentiment polarity and subjectivity
        """
        blob = TextBlob(text)
        
        # Get polarity score (-1 to 1)
        polarity = blob.sentiment.polarity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment': sentiment
        }
    
    def analyze_comments_batch(self, comments_df):
        """
        Analyze sentiment for a batch of comments
        """
        results = []
        
        for idx, row in comments_df.iterrows():
            analysis = self.analyze_sentiment(row['comment_text'])
            results.append({
                'comment_id': row['comment_id'],
                'stakeholder_name': row['stakeholder_name'],
                'comment_text': row['comment_text'],
                'provision_reference': row['provision_reference'],
                'sentiment': analysis['sentiment'],
                'polarity_score': round(analysis['polarity'], 3),
                'subjectivity_score': round(analysis['subjectivity'], 3)
            })
        
        return pd.DataFrame(results)
    
    def get_overall_sentiment(self, sentiments_df):
        """
        Calculate overall sentiment statistics
        """
        sentiment_counts = sentiments_df['sentiment'].value_counts().to_dict()
        avg_polarity = sentiments_df['polarity_score'].mean()
        
        # Determine overall sentiment
        if avg_polarity > 0.1:
            overall_sentiment = 'Positive'
        elif avg_polarity < -0.1:
            overall_sentiment = 'Negative'
        else:
            overall_sentiment = 'Neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'average_polarity': round(avg_polarity, 3),
            'sentiment_distribution': sentiment_counts,
            'total_comments': len(sentiments_df),
            'positive_percentage': round(sentiment_counts.get('Positive', 0) / len(sentiments_df) * 100, 1),
            'negative_percentage': round(sentiment_counts.get('Negative', 0) / len(sentiments_df) * 100, 1),
            'neutral_percentage': round(sentiment_counts.get('Neutral', 0) / len(sentiments_df) * 100, 1)
        }
    
    def extract_keywords(self, text, num_keywords=10):
        """
        Extract most frequent keywords from text
        """
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Split into words
        words = text.split()
        
        # Remove stop words and short words
        keywords = [word for word in words if word not in self.stop_words and len(word) > 3]
        
        # Count frequency
        word_freq = Counter(keywords)
        
        # Return top keywords
        return word_freq.most_common(num_keywords)
    
    def get_provision_wise_sentiment(self, sentiments_df):
        """
        Group sentiments by provision reference
        """
        provision_sentiment = {}
        
        for provision in sentiments_df['provision_reference'].unique():
            provision_data = sentiments_df[sentiments_df['provision_reference'] == provision]
            provision_sentiment[provision] = {
                'count': len(provision_data),
                'avg_polarity': round(provision_data['polarity_score'].mean(), 3),
                'sentiments': provision_data['sentiment'].value_counts().to_dict()
            }
        
        return provision_sentiment
    
    def analyze(self, df):
        """
        Main analysis method that combines all sentiment analysis functions
        Returns a dictionary with summary, detailed results, and breakdowns
        """
        # Perform batch sentiment analysis
        detailed_results = self.analyze_comments_batch(df)
        
        # Get overall sentiment statistics
        overall_stats = self.get_overall_sentiment(detailed_results)
        
        # Get provision-wise sentiment
        provision_sentiment = self.get_provision_wise_sentiment(detailed_results)
        
        # Get stakeholder-wise sentiment
        stakeholder_sentiment = {}
        for stakeholder in detailed_results['stakeholder_name'].unique():
            stakeholder_data = detailed_results[detailed_results['stakeholder_name'] == stakeholder]
            stakeholder_sentiment[stakeholder] = {
                'positive': len(stakeholder_data[stakeholder_data['sentiment'] == 'Positive']),
                'negative': len(stakeholder_data[stakeholder_data['sentiment'] == 'Negative']),
                'neutral': len(stakeholder_data[stakeholder_data['sentiment'] == 'Neutral']),
                'total': len(stakeholder_data)
            }
        
        # Convert sentiment distribution keys to lowercase for consistency
        sentiment_summary = {}
        for key, value in overall_stats['sentiment_distribution'].items():
            sentiment_summary[key.lower()] = value
        
        return {
            'summary': sentiment_summary,
            'detailed': detailed_results,
            'by_provision': provision_sentiment,
            'by_stakeholder': stakeholder_sentiment,
            'overall_stats': overall_stats
        }
