import unittest
from unittest.mock import patch, MagicMock
from app import app
import json

class TestSentimentAnalysisApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    @patch('app.collection.insert_one')
    def test_submit_sentence(self, mock_insert_one):
        """Test the /checkSentiment endpoint."""
        # Mock the database insertion
        mock_insert_one.return_value = MagicMock(inserted_id="mock_id")

        # Define test payload
        payload = {"sentence": "This is a test. This is another test."}

        # Make POST request
        response = self.client.post(
            '/checkSentiment',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Assert response
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("request_id", data)

    def test_submit_empty_sentence(self):
        """Test the /checkSentiment endpoint with empty input."""
        payload = {"sentence": ""}
        response = self.client.post(
            '/checkSentiment',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Input text is empty.")

    @patch('app.collection.find_one')
    def test_get_analysis_processed(self, mock_find_one):
        """Test the /get_analysis endpoint with processed status."""
        # Mock a processed document
        mock_find_one.return_value = {
            "request_id": "mock_request_id",
            "overall_status": "processed",
            "sentences": [{"sentence": "This is a test.", "status": "processed", "analysis": "positive"}]
        }

        response = self.client.get('/get_analysis?request_id=mock_request_id')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("request_id", data)
        self.assertEqual(data["overall_status"], "processed")

    @patch('app.collection.find_one')
    def test_get_analysis_not_found(self, mock_find_one):
        """Test the /get_analysis endpoint with no document found."""
        mock_find_one.return_value = None

        response = self.client.get('/get_analysis?request_id=nonexistent_request_id')

        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn("message", data)
        self.assertEqual(data["message"], "No analysis found")

    @patch('app.collection.find_one')
    def test_get_emotion_intensity(self, mock_find_one):
        """Test the /emotion_intensity/<string:request_id> endpoint."""
        # Mock a document with emotion data
        mock_find_one.return_value = {
            "request_id": "mock_request_id",
            "sentences": [
                {"sentence": "This is a happy sentence.", "emotions": ["happy"]},
                {"sentence": "This is a sad sentence.", "emotions": ["sad"]},
                {"sentence": "Another happy one.", "emotions": ["happy"]}
            ]
        }

        response = self.client.get('/emotion_intensity/mock_request_id')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("happy", data)
        self.assertIn("sad", data)
        self.assertEqual(data["happy"], 2/3)
        self.assertEqual(data["sad"], 1/3)

    @patch('app.collection.find_one')
    def test_get_emotion_intensity_not_found(self, mock_find_one):
        """Test the /emotion_intensity/<string:request_id> endpoint with no document found."""
        mock_find_one.return_value = None

        response = self.client.get('/emotion_intensity/nonexistent_request_id')

        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Document not found")

if __name__ == '__main__':
    unittest.main()
