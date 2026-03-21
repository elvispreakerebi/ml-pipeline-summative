"""
Locust load testing for the Emergency Call Emotion Classifier API.

Usage:
    locust -f locust/locustfile.py --host=http://localhost:8000

    # Headless mode (for CI or recording results):
    locust -f locust/locustfile.py --host=http://localhost:8000 \
        --headless -u 50 -r 10 --run-time 60s --csv=locust/results/report

    # Test with different Docker container counts:
    docker-compose up --scale api=1   # 1 container
    docker-compose up --scale api=2   # 2 containers
    docker-compose up --scale api=4   # 4 containers
"""

import os
from locust import HttpUser, task, between

# Path to a sample WAV file for testing predictions
SAMPLE_WAV = os.path.join(os.path.dirname(__file__), "sample_test.wav")


class EmotionClassifierUser(HttpUser):
    """Simulates a user interacting with the emotion classifier API."""

    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        """Check API health status."""
        self.client.get("/health")

    @task(2)
    def get_metrics(self):
        """Fetch current model metrics."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 404:
                response.success()  # Expected if no metrics yet

    @task(2)
    def get_classes(self):
        """Fetch available emotion classes."""
        self.client.get("/classes")

    @task(1)
    def get_class_distribution(self):
        """Fetch dataset class distribution."""
        with self.client.get("/insights/class-distribution", catch_response=True) as response:
            if response.status_code == 404:
                response.success()

    @task(3)
    def predict_emotion(self):
        """Upload a WAV file and get emotion prediction."""
        if not os.path.exists(SAMPLE_WAV):
            # Generate a minimal WAV file if sample doesn't exist
            self._generate_sample_wav()

        with open(SAMPLE_WAV, "rb") as f:
            self.client.post(
                "/predict",
                files={"file": ("test.wav", f, "audio/wav")},
            )

    @task(1)
    def predict_with_spectrogram(self):
        """Upload WAV and get prediction with spectrogram."""
        if not os.path.exists(SAMPLE_WAV):
            self._generate_sample_wav()

        with open(SAMPLE_WAV, "rb") as f:
            self.client.post(
                "/predict/spectrogram",
                files={"file": ("test.wav", f, "audio/wav")},
            )

    def _generate_sample_wav(self):
        """Generate a minimal WAV file for testing."""
        import struct
        import wave

        sample_rate = 22050
        duration = 3  # seconds
        num_samples = sample_rate * duration

        os.makedirs(os.path.dirname(SAMPLE_WAV), exist_ok=True)

        with wave.open(SAMPLE_WAV, "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)

            # Generate a simple sine wave
            import math
            for i in range(num_samples):
                value = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
                wav.writeframes(struct.pack("<h", value))
