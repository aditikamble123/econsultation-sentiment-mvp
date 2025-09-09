"""
Text Summarization Module for E-consultation Comments
"""

import re
from collections import Counter
import pandas as pd
from textblob import TextBlob

class TextSummarizer:
    def __init__(self):
        self.stop_words = set(['the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'in', 
                               'to', 'for', 'of', 'with', 'as', 'by', 'that', 'this', 
                               'it', 'from', 'or', 'but', 'are', 'was', 'were', 'be'])
    
    def summarize_single_comment(self, text, max_length=50):
        """
        Create a brief summary of a single comment
        """
        # If text is already short, return as is
        words = text.split()
        if len(words) <= max_length:
            return text
        
        # Extract key sentences based on word importance
        sentences = text.split('.')
        if len(sentences) <= 2:
            # For very short texts, just truncate
            return ' '.join(words[:max_length]) + '...'
        
        # Score sentences based on word frequency
        word_freq = self._get_word_frequency(text)
        sentence_scores = {}
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 0:
                words_in_sentence = sentence.lower().split()
                score = sum(word_freq.get(word, 0) for word in words_in_sentence)
                sentence_scores[sentence] = score / len(words_in_sentence) if words_in_sentence else 0
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        summary = '. '.join([sent[0] for sent in top_sentences])
        
        # Ensure summary is not too long
        summary_words = summary.split()
        if len(summary_words) > max_length:
            summary = ' '.join(summary_words[:max_length]) + '...'
        
        return summary
    
    def _get_word_frequency(self, text):
        """
        Calculate word frequency for text
        """
        # Clean text
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = [word for word in text.split() if word not in self.stop_words and len(word) > 2]
        
        # Calculate frequency
        word_freq = Counter(words)
        
        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        return word_freq
    
    def generate_overall_summary(self, comments_df, sentiments_df):
        """
        Generate an overall summary of all comments
        """
        total_comments = len(comments_df)
        
        # Get sentiment distribution
        sentiment_counts = sentiments_df['sentiment'].value_counts()
        
        # Find most discussed provisions
        provision_counts = comments_df['provision_reference'].value_counts()
        top_provisions = provision_counts.head(3).to_dict()
        
        # Extract key themes
        all_text = ' '.join(comments_df['comment_text'].tolist())
        word_freq = self._get_word_frequency(all_text)
        top_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        top_theme_words = [theme[0] for theme in top_themes]
        
        # Generate summary
        summary = f"Analysis of {total_comments} stakeholder comments reveals:\n\n"
        
        # Sentiment summary
        summary += "SENTIMENT OVERVIEW:\n"
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_comments) * 100
            summary += f"• {sentiment}: {count} comments ({percentage:.1f}%)\n"
        
        # Key provisions
        summary += f"\nMOST DISCUSSED PROVISIONS:\n"
        for provision, count in top_provisions.items():
            summary += f"• {provision}: {count} comments\n"
        
        # Key themes
        summary += f"\nKEY THEMES:\n"
        summary += f"• Main topics: {', '.join(top_theme_words)}\n"
        
        # Key insights based on sentiment
        positive_comments = sentiments_df[sentiments_df['sentiment'] == 'Positive']
        negative_comments = sentiments_df[sentiments_df['sentiment'] == 'Negative']
        
        if len(positive_comments) > 0:
            summary += "\nPOSITIVE FEEDBACK:\n"
            # Get common positive words
            positive_text = ' '.join(positive_comments['comment_text'].tolist())
            positive_keywords = self._extract_sentiment_keywords(positive_text, 3)
            summary += f"• Stakeholders appreciate: {', '.join(positive_keywords)}\n"
        
        if len(negative_comments) > 0:
            summary += "\nCONCERNS RAISED:\n"
            # Get common negative words
            negative_text = ' '.join(negative_comments['comment_text'].tolist())
            negative_keywords = self._extract_sentiment_keywords(negative_text, 3)
            summary += f"• Main concerns: {', '.join(negative_keywords)}\n"
        
        return summary
    
    def _extract_sentiment_keywords(self, text, num_keywords=5):
        """
        Extract keywords specific to sentiment
        """
        # Clean and tokenize
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = [word for word in text.split() if word not in self.stop_words and len(word) > 3]
        
        # Filter for meaningful words (nouns, verbs, adjectives)
        meaningful_words = []
        for word in words:
            if word in ['compliance', 'regulation', 'business', 'cost', 'implementation', 
                       'transparency', 'governance', 'reform', 'burden', 'support', 
                       'accountability', 'framework', 'amendment', 'provision', 
                       'requirement', 'documentation', 'innovation', 'growth']:
                meaningful_words.append(word)
        
        # Get most common
        word_freq = Counter(meaningful_words)
        top_words = word_freq.most_common(num_keywords)
        
        return [word[0] for word in top_words]
    
    def create_stakeholder_summary(self, comments_df, sentiments_df):
        """
        Create a summary grouped by stakeholder type
        """
        # Group stakeholders by patterns in their names
        stakeholder_groups = {
            'Business': [],
            'Legal': [],
            'Government': [],
            'Individual': [],
            'Association': []
        }
        
        for idx, row in comments_df.iterrows():
            name = row['stakeholder_name'].lower()
            sentiment_row = sentiments_df[sentiments_df['comment_id'] == row['comment_id']].iloc[0]
            
            if 'firm' in name or 'legal' in name or 'lawyer' in name:
                stakeholder_groups['Legal'].append(sentiment_row)
            elif 'business' in name or 'company' in name or 'ceo' in name or 'founder' in name:
                stakeholder_groups['Business'].append(sentiment_row)
            elif 'ministry' in name or 'government' in name:
                stakeholder_groups['Government'].append(sentiment_row)
            elif 'association' in name or 'chamber' in name or 'group' in name or 'org' in name:
                stakeholder_groups['Association'].append(sentiment_row)
            else:
                stakeholder_groups['Individual'].append(sentiment_row)
        
        # Create summary for each group
        group_summary = {}
        for group, comments in stakeholder_groups.items():
            if comments:
                comments_df_group = pd.DataFrame(comments)
                sentiment_dist = comments_df_group['sentiment'].value_counts().to_dict()
                avg_polarity = comments_df_group['polarity_score'].mean()
                
                group_summary[group] = {
                    'count': len(comments),
                    'sentiment_distribution': sentiment_dist,
                    'average_polarity': round(avg_polarity, 3)
                }
        
        return group_summary
    
    def summarize(self, df, sentiments_df):
        """
        Main summarization method that combines all text summarization functions
        Returns a dictionary with overall summary, key themes, and provision summaries
        """
        # Generate overall summary
        overall_summary = self.generate_overall_summary(df, sentiments_df)
        
        # Extract key themes
        all_text = ' '.join(df['comment_text'].tolist())
        word_freq = self._get_word_frequency(all_text)
        top_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        key_themes = [theme[0] for theme in top_themes]
        
        # Generate provision-wise summaries
        provision_summaries = {}
        for provision in df['provision_reference'].unique():
            provision_comments = df[df['provision_reference'] == provision]
            provision_text = ' '.join(provision_comments['comment_text'].tolist())
            provision_summaries[provision] = self.summarize_single_comment(provision_text, max_length=100)
        
        # Get stakeholder group summary
        stakeholder_summary = self.create_stakeholder_summary(df, sentiments_df)
        
        return {
            'overall_summary': overall_summary,
            'key_themes': key_themes,
            'provision_summaries': provision_summaries,
            'stakeholder_groups': stakeholder_summary
        }
