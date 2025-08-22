
import requests

url = "http://localhost:8000/acquire_image"
params = {"filename": "test_image", "image_side_length": 512}

response = requests.get(url, params=params)
data = response.json()
print(data)