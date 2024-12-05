"""
Unit tests for the app's main functionality.
"""

import os
from unittest.mock import patch, MagicMock
import pytest
import nltk
from bson import ObjectId
from datetime import datetime

from app import (
    perform_sentiment_analysis,
    perform_topic_modeling,
    perform_emotion_detection,
    perform_text_summarization,
    perform_sentiment_trend_analysis,
    perform_overall_emotion_detection,
    process_document,
    perform_ner,
    update_document_in_db,
)

def test_perform_ner():
    """Test the NER function."""
    sample_sentences = [
        {"sentence": "Barack Obama was the 44th President of the United States."},
        {"sentence": "Google was founded in September 1998 by Larry Page and Sergey Brin."}
    ]
    result = perform_ner(sample_sentences)
    assert len(result) == 2
    assert "entities" in result[0]
    assert result[0]["entities"] == [
        {"text": "Barack Obama", "label": "PERSON"},
        {"text": "44th", "label": "ORDINAL"},
        {"text": "the United States", "label": "GPE"}
    ]

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up testing environment variable."""
    os.environ["TESTING"] = "true"
    yield
    del os.environ["TESTING"]


@pytest.fixture(scope="session", autouse=True)
def setup_nltk_resources():
    """Set up necessary NLTK resources."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


@pytest.fixture
def sample_sentences_small_fixture():
    """Sample data with 2 sentences."""
    return [
        {
            "sentence": "This is a happy sentence.",
            "status": "processed",
            "analysis": {"compound": 0.5, "neg": 0.0, "neu": 0.3, "pos": 0.7},
        },
        {
            "sentence": "This is a sad sentence.",
            "status": "processed",
            "analysis": {"compound": -0.5, "neg": 0.7, "neu": 0.3, "pos": 0.0},
        },
    ]


@pytest.fixture
def sample_sentences_large_fixture():
    """Sample data with varied sentences to satisfy LDA requirements."""
    return [
        {
            "sentence": "This is a happy sentence about technology and innovation.",
            "status": "processed",
            "analysis": {"compound": 0.5, "neg": 0.0, "neu": 0.3, "pos": 0.7},
        },
        {
            "sentence": "The weather today is quite rainy and cold.",
            "status": "processed",
            "analysis": {"compound": -0.3, "neg": 0.4, "neu": 0.6, "pos": 0.0},
        },
        {
            "sentence": "Machine learning algorithms are processing data efficiently.",
            "status": "processed",
            "analysis": {"compound": 0.4, "neg": 0.0, "neu": 0.6, "pos": 0.4},
        },
        {
            "sentence": "The new software update improves system performance.",
            "status": "processed",
            "analysis": {"compound": 0.6, "neg": 0.0, "neu": 0.4, "pos": 0.6},
        },
        {
            "sentence": "Users reported issues with the latest feature.",
            "status": "processed",
            "analysis": {"compound": -0.4, "neg": 0.5, "neu": 0.5, "pos": 0.0},
        },
    ] * 4  # Creates 20 items


def test_perform_sentiment_analysis(
    sample_sentences_small_fixture,
):  # pylint: disable=W0621
    """Test the sentiment analysis function."""
    result = perform_sentiment_analysis(sample_sentences_small_fixture)
    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    assert result[0]["status"] == "processed"
    assert "compound" in result[0]["analysis"]
    assert "neg" in result[0]["analysis"]
    assert "neu" in result[0]["analysis"]
    assert "pos" in result[0]["analysis"]


def test_perform_topic_modeling(
    sample_sentences_large_fixture,
):  # pylint: disable=W0621
    """Test the topic modeling function."""
    with patch("app.Phrases") as _mock_phrases, patch(
        "app.Phraser"
    ) as _mock_phraser, patch("gensim.corpora.Dictionary") as mock_dictionary, patch(
        "gensim.models.LdaModel"
    ) as mock_lda_model:
        mock_dict = MagicMock()
        mock_dict.filter_extremes.return_value = None
        mock_dict.__len__.return_value = 10
        mock_dict.doc2bow = MagicMock(return_value=[(0, 1), (1, 1)])
        mock_dictionary.return_value = mock_dict

        mock_lda = MagicMock()
        mock_lda.print_topics.return_value = [
            (0, "word1 0.1 word2 0.09"),
            (1, "word3 0.08 word4 0.07"),
        ]
        mock_lda_model.return_value = mock_lda

        with patch("app.CoherenceModel", side_effect=Exception("Should not be called")):
            topics = perform_topic_modeling(
                sample_sentences_large_fixture, num_topics=2
            )
            assert isinstance(topics, list), "Topics should be a list"
            assert len(topics) == 2, f"Expected 2 topics, got {len(topics)}"
            assert all(
                isinstance(topic, tuple) for topic in topics
            ), "Each topic should be a tuple"


def test_perform_emotion_detection(
    sample_sentences_small_fixture,
):  # pylint: disable=W0621
    """Test the emotion detection function."""
    result = perform_emotion_detection(sample_sentences_small_fixture)
    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    assert "emotions" in result[0], "Missing 'emotions' key in the first result"
    assert isinstance(result[0]["emotions"], list), "'emotions' should be a list"


def test_perform_text_summarization(
    sample_sentences_large_fixture,
):  # pylint: disable=W0621
    """Test the text summarization function."""
    summary = perform_text_summarization(sample_sentences_large_fixture)
    assert isinstance(summary, str), "Summary should be a string"
    assert len(summary) > 0, "Summary should not be empty"


def test_perform_sentiment_trend_analysis(
    sample_sentences_small_fixture,
):  # pylint: disable=W0621
    """Test the sentiment trend analysis function."""
    result = perform_sentiment_trend_analysis(sample_sentences_small_fixture)
    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    assert (
        "sentence_index" in result[0]
    ), "Missing 'sentence_index' key in the first result"
    assert "compound" in result[0], "Missing 'compound' key in the first result"


def test_perform_overall_emotion_detection():  # No fixture involved
    """Test the overall emotion detection function."""
    sentences_with_emotions = [
        {"sentence": "This is a happy sentence.", "emotions": ["Happy"]},
        {"sentence": "This is a sad sentence.", "emotions": ["Sad"]},
        {"sentence": "This is also a happy sentence.", "emotions": ["Happy"]},
    ]
    result = perform_overall_emotion_detection(sentences_with_emotions)
    assert isinstance(result, list), "Overall emotions should be a list"
    assert len(result) > 0, "Overall emotions list should not be empty"
    assert "Happy" in result, "'Happy' should be in the overall emotions"


def test_process_document(sample_sentences_large_fixture):  # pylint: disable=W0621
    """Test the document processing function."""
    sample_document = {
        "_id": "1234567890",
        "request_id": "unique_request_id",
        "sentences": sample_sentences_large_fixture,
        "overall_status": "pending",
        "timestamp": "2024-11-13T04:00:00",
    }
    with patch("app.perform_sentiment_analysis") as mock_sentiment, patch(
        "app.perform_topic_modeling"
    ) as mock_topic_modeling, patch(
        "app.perform_emotion_detection"
    ) as mock_emotion_detection, patch(
        "app.perform_text_summarization"
    ) as mock_summary, patch(
        "app.perform_sentiment_trend_analysis"
    ) as mock_sentiment_trend, patch(
        "app.perform_overall_emotion_detection"
    ) as mock_overall_emotion:

        mock_sentiment.return_value = sample_sentences_large_fixture
        mock_topic_modeling.return_value = ["Topic1", "Topic2"]
        mock_emotion_detection.return_value = sample_sentences_large_fixture
        mock_summary.return_value = "This is a summary."
        mock_sentiment_trend.return_value = [
            {"sentence_index": 0, "compound": 0.5},
            {"sentence_index": 1, "compound": -0.5},
            # Add additional mocked trend data if necessary
        ]
        mock_overall_emotion.return_value = ["Happy"]

        processed_document = process_document(sample_document)
        assert (
            processed_document["overall_status"] == "processed"
        ), "Overall status should be 'processed'"
        assert (
            "topics" in processed_document
        ), "Missing 'topics' in the processed document"
        assert (
            "summary" in processed_document
        ), "Missing 'summary' in the processed document"
        assert (
            "sentiment_trend" in processed_document
        ), "Missing 'sentiment_trend' in the processed document"
        assert (
            "overall_emotions" in processed_document
        ), "Missing 'overall_emotions' in the processed document"

def test_update_document_in_db():
    """Test updating a document in the database."""
    sample_document = {
        "_id": ObjectId("507f1f77bcf86cd799439011"),
        "sentences": [
            {"sentence": "I am happy.", "analysis": {"compound": 0.5}, "emotions": ["Happy"], "entities": []},
            {"sentence": "I am sad.", "analysis": {"compound": -0.5}, "emotions": ["Sad"], "entities": []}
        ],
        "overall_status": "processed",
        "topics": ["Topic1", "Topic2"],
        "summary": "This is a summary.",
        "sentiment_trend": [{"sentence_index": 0, "compound": 0.5}, {"sentence_index": 1, "compound": -0.5}],
        "overall_emotions": ["Happy"],
        "timestamp": datetime.now()
    }

    with patch('app.texts_collection.update_one') as mock_update:
        update_document_in_db(sample_document)
        mock_update.assert_called_once_with(
            {"_id": sample_document["_id"]},
            {"$set": {
                "sentences": sample_document["sentences"],
                "overall_status": sample_document["overall_status"],
                "topics": sample_document["topics"],
                "summary": sample_document["summary"],
                "sentiment_trend": sample_document["sentiment_trend"],
                "overall_emotions": sample_document["overall_emotions"],
                "timestamp": sample_document["timestamp"],
            }}
        )

def test_perform_topic_modeling_no_sentences():
    """Test topic modeling with no sentences."""
    result = perform_topic_modeling([], num_topics=5)
    assert result == [], "Expected an empty list when no sentences are provided."

def test_perform_topic_modeling_with_valid_data(sample_sentences_large_fixture):
    """Test topic modeling with valid data, ensuring mocks return appropriate values."""
    # Mock external dependencies if necessary
    with patch('app.corpora.Dictionary') as mock_dictionary:
        mock_dict = MagicMock()
        mock_dict.__len__.return_value = 10  # Ensure dictionary length is non-zero
        mock_dict.doc2bow.return_value = [(0, 1), (1, 2)]
        mock_dictionary.return_value = mock_dict

        with patch('app.models.LdaModel') as mock_lda_model:
            mock_lda = MagicMock()
            mock_lda.print_topics.return_value = [
                (0, "0.1*'word1' + 0.09*'word2'"),
                (1, "0.08*'word3' + 0.07*'word4'")
            ]
            mock_lda_model.return_value = mock_lda

            topics = perform_topic_modeling(sample_sentences_large_fixture, num_topics=2)
            assert isinstance(topics, list)
            assert len(topics) == 2
            assert isinstance(topics[0], tuple)
            assert isinstance(topics[1], tuple)
            mock_dictionary.assert_called_once()
            mock_lda_model.assert_called_once()
            mock_lda.print_topics.assert_called_once_with(num_words=4)



def test_update_document_in_db_processed():
    """Test updating a document with 'processed' status."""
    sample_document = {
        "_id": ObjectId("507f1f77bcf86cd799439011"),
        "sentences": [
            {"sentence": "I am happy.", "analysis": {"compound": 0.5}, "emotions": ["Happy"], "entities": []},
            {"sentence": "I am sad.", "analysis": {"compound": -0.5}, "emotions": ["Sad"], "entities": []}
        ],
        "overall_status": "processed",
        "topics": ["Topic1", "Topic2"],
        "summary": "This is a summary.",
        "sentiment_trend": [{"sentence_index": 0, "compound": 0.5}, {"sentence_index": 1, "compound": -0.5}],
        "overall_emotions": ["Happy"],
        "timestamp": datetime.now()
    }

    with patch('app.texts_collection.update_one') as mock_update:
        update_document_in_db(sample_document)
        mock_update.assert_called_once_with(
            {"_id": sample_document["_id"]},
            {"$set": {
                "sentences": sample_document["sentences"],
                "overall_status": sample_document["overall_status"],
                "topics": sample_document["topics"],
                "summary": sample_document["summary"],
                "sentiment_trend": sample_document["sentiment_trend"],
                "overall_emotions": sample_document["overall_emotions"],
                "timestamp": sample_document["timestamp"],
            }}
        )           

def test_perform_topic_modeling_empty_sentences():
    """Test topic modeling with empty sentences."""
    result = perform_topic_modeling([], num_topics=5)
    assert result == [], "Expected an empty list when empty sentences are provided."

def test_perform_sentiment_analysis_neutral():
    """Test sentiment analysis with a neutral sentence."""
    sample_sentences = [
        {
            "sentence": "The sky is blue.",
            "status": "pending",
            "analysis": None,
        }
    ]
    result = perform_sentiment_analysis(sample_sentences)
    assert result[0]["analysis"]["compound"] == 0.0  # Assuming neutral compound score
    assert result[0]["status"] == "processed"

def test_perform_topic_modeling_valid_data(sample_sentences_large_fixture):
    """Test topic modeling with valid data."""
    topics = perform_topic_modeling(sample_sentences_large_fixture, num_topics=2)
    assert isinstance(topics, list)
    assert len(topics) == 2
    for topic in topics:
        assert isinstance(topic, tuple)
        assert isinstance(topic[0], int)
        assert isinstance(topic[1], str)

@pytest.fixture
def sample_document_processed():
    return {
        "_id": ObjectId("507f1f77bcf86cd799439011"),
        "request_id": "unique_request_id",
        "sentences": [
            {"sentence": "I am happy.", "analysis": {"compound": 0.5}, "emotions": ["Happy"], "entities": []},
            {"sentence": "I am sad.", "analysis": {"compound": -0.5}, "emotions": ["Sad"], "entities": []}
        ],
        "overall_status": "processed",
        "topics": ["Topic1", "Topic2"],
        "summary": "This is a summary.",
        "sentiment_trend": [{"sentence_index": 0, "compound": 0.5}, {"sentence_index": 1, "compound": -0.5}],
        "overall_emotions": ["Happy"],
        "timestamp": datetime.now()
    }

def test_update_document_in_db_error():
    """Test updating a document when the database update fails."""
    sample_document = {
        "_id": ObjectId("507f1f77bcf86cd799439011"),
        "overall_status": "processed",
        "timestamp": datetime.now(),
    }

    with patch('app.texts_collection.update_one', side_effect=Exception("Database Error")) as mock_update:
        with pytest.raises(Exception) as excinfo:
            update_document_in_db(sample_document)
        assert "Database Error" in str(excinfo.value)
        mock_update.assert_called_once()

def test_process_document_no_sentences():
    """Test processing a document with no sentences."""
    sample_document = {
        "_id": "1234567890",
        "request_id": "unique_request_id",
        "sentences": [],
        "overall_status": "pending",
        "timestamp": "2024-11-13T04:00:00",
    }
    processed_document = process_document(sample_document)
    assert processed_document["overall_status"] == "error"
    assert "error_message" in processed_document
    assert processed_document["error_message"] == "No sentences to process."

def test_perform_ner_no_entities():
    """Test the NER function with sentences that have no named entities."""
    sample_sentences = [
        {"sentence": "This is a test sentence without entities."},
        {"sentence": "Another simple sentence."}
    ]
    result = perform_ner(sample_sentences)
    assert len(result) == 2
    assert "entities" in result[0]
    assert result[0]["entities"] == []
    assert result[1]["entities"] == []
