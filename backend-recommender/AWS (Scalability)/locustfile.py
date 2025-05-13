from locust import HttpUser, task, between

class LoadTest(HttpUser):
    wait_time = between(1, 3)

    @task
    def recommend(self):
        headers = {"Content-Type": "application/json"}
        payload = {
            "spotify_url": "https://open.spotify.com/track/5CQ30WqJwcep0pYcV4AMNc?si=810aafbc65b1413d"
        }
        self.client.post("/api/recommend", json=payload, headers=headers)