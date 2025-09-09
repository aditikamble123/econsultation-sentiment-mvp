"""
Word Cloud Generation Module for E-consultation Comments
"""

import base64
from io import BytesIO
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for web deployment
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import Counter

class WordCloudGenerator:
    def __init__(self):
        self.stop_words = set(['the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'in', 
                               'to', 'for', 'of', 'with', 'as', 'by', 'that', 'this', 
                               'it', 'from', 'or', 'but', 'are', 'was', 'were', 'be',
                               'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                               'could', 'should', 'may', 'might', 'must', 'can', 'shall',
                               'we', 'our', 'us', 'them', 'they', 'their', 'these', 'those',
                               'been', 'being', 'having', 'more', 'very', 'some', 'any'])
    
    def generate_wordcloud(self, text, max_words=50):
        """
        Generate word cloud from text
        """
        # Clean text
        text = self._clean_text(text)
        
        # Create WordCloud object
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=self.stop_words,
            max_words=max_words,
            relative_scaling=0.5,
            colormap='viridis'
        ).generate(text)
        
        # Generate image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.title('Word Cloud - Stakeholder Comments', fontsize=16, pad=20)
        plt.tight_layout(pad=0)
        
        # Convert to base64 for web display
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def generate_sentiment_wordclouds(self, sentiments_df):
        """
        Generate separate word clouds for positive and negative sentiments
        """
        wordclouds = {}
        
        # Positive comments word cloud
        positive_comments = sentiments_df[sentiments_df['sentiment'] == 'Positive']
        if len(positive_comments) > 0:
            positive_text = ' '.join(positive_comments['comment_text'].tolist())
            wordclouds['positive'] = self._generate_colored_wordcloud(
                positive_text, 'Greens', 'Positive Sentiment Word Cloud'
            )
        
        # Negative comments word cloud
        negative_comments = sentiments_df[sentiments_df['sentiment'] == 'Negative']
        if len(negative_comments) > 0:
            negative_text = ' '.join(negative_comments['comment_text'].tolist())
            wordclouds['negative'] = self._generate_colored_wordcloud(
                negative_text, 'Reds', 'Negative Sentiment Word Cloud'
            )
        
        # Neutral comments word cloud
        neutral_comments = sentiments_df[sentiments_df['sentiment'] == 'Neutral']
        if len(neutral_comments) > 0:
            neutral_text = ' '.join(neutral_comments['comment_text'].tolist())
            wordclouds['neutral'] = self._generate_colored_wordcloud(
                neutral_text, 'Blues', 'Neutral Sentiment Word Cloud'
            )
        
        return wordclouds
    
    def _generate_colored_wordcloud(self, text, colormap, title):
        """
        Generate word cloud with specific color scheme
        """
        text = self._clean_text(text)
        
        wordcloud = WordCloud(
            width=600,
            height=300,
            background_color='white',
            stopwords=self.stop_words,
            max_words=30,
            relative_scaling=0.5,
            colormap=colormap
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.title(title, fontsize=14, pad=15)
        plt.tight_layout(pad=0)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _clean_text(self, text):
        """
        Clean text for word cloud generation
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_word_frequency(self, text, top_n=20):
        """
        Get word frequency distribution
        """
        # Clean text
        text = self._clean_text(text)
        
        # Split into words
        words = text.split()
        
        # Remove stop words
        words = [word for word in words if word not in self.stop_words and len(word) > 3]
        
        # Count frequency
        word_freq = Counter(words)
        
        # Get top N words
        top_words = word_freq.most_common(top_n)
        
        return top_words
    
    def generate_provision_wordcloud(self, comments_df, provision):
        """
        Generate word cloud for specific provision
        """
        # Filter comments for the provision
        provision_comments = comments_df[comments_df['provision_reference'] == provision]
        
        if len(provision_comments) == 0:
            return None
        
        # Combine all comments
        text = ' '.join(provision_comments['comment_text'].tolist())
        
        # Generate word cloud
        return self.generate_wordcloud(text, max_words=30)
    
    def create_frequency_chart_data(self, comments_df):
        """
        Create data for frequency chart visualization
        """
        # Get all text
        all_text = ' '.join(comments_df['comment_text'].tolist())
        
        # Get word frequency
        word_freq = self.get_word_frequency(all_text, top_n=15)
        
        # Prepare data for chart
        words = [item[0] for item in word_freq]
        frequencies = [item[1] for item in word_freq]
        
        return {
            'words': words,
            'frequencies': frequencies
        }
    
    def generate(self, df, sentiments_df):
        """
        Main generation method that combines all word cloud generation functions
        Returns a dictionary with main word cloud and sentiment-based word clouds
        """
        # Generate main word cloud from all comments
        all_text = ' '.join(df['comment_text'].tolist())
        main_wordcloud = self.generate_wordcloud(all_text, max_words=50)
        
        # Generate sentiment-based word clouds
        sentiment_wordclouds = self.generate_sentiment_wordclouds(sentiments_df)
        
        # Get word frequency data for charts
        frequency_data = self.create_frequency_chart_data(df)
        
        # Generate provision-specific word clouds for top provisions
        provision_counts = df['provision_reference'].value_counts()
        top_provisions = provision_counts.head(3).index.tolist()
        provision_wordclouds = {}
        for provision in top_provisions:
            provision_wordclouds[provision] = self.generate_provision_wordcloud(df, provision)
        
        return {
            'main_wordcloud': main_wordcloud,
            'sentiment_wordclouds': sentiment_wordclouds,
            'provision_wordclouds': provision_wordclouds,
            'frequency_data': frequency_data
        }
