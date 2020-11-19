# coin_detector_backend

This is purely an API which receives an image and identifies the coins present in the image.

## Request format

This API accepts POST request which should be in JSON format as given below,

`{
  'image' : base64 encoded string of the image
}`
