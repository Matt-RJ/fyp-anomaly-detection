import requests
import os
import random
import mimetypes

api_url = 'https://fhod7w2r3i.execute-api.eu-west-1.amazonaws.com/v1/Image/Resize'

def pick_random_file(dir_path):
  dir_path = os.path.join(os.path.dirname(__file__), dir_path)
  return random.choice(os.listdir(dir_path))

def hit_image_processor(image_filepath):
  image_filepath = os.path.join(os.path.dirname(__file__), image_filepath)
  image = open(image_filepath, 'rb')
  mime = mimetypes.guess_type(image_filepath, strict=True)[0]
  return requests.post(
    url=api_url,
    data={
      'ResizeX': 500,
      'ResizeY': 500
    },
    files=[('Image', (image_filepath, image, mime))]
  )

i = 0
while True:
  image = pick_random_file('images')
  print(f'Run {i}. Using {image}.')
  res = hit_image_processor(f'images\{pick_random_file("images")}')
  if (res.ok):
    print(res.status_code, 'OK')
  else:
    print(res.status_code, res.text)
  print()
  i += 1