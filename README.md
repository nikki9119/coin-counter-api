# coin_detector_backend

This is purely an API which receives an image and identifies the coins present in the image.

## Request format

The request must be sent to the `\predict` route.

The API accepts POST request which should be in JSON format as given below,

`{
  'image' : base64 encoded string of the image
}`

The image should be encoded to a base64 format string for transmission. You can encode your image using *base64* module in python or use this url

https://base64.guru/converter/encode/image

## Response format

The API sends the response of the following JSON format,

```json
{
  "preds":"ok",
  "results":{   
    "one": number of one rupee coins,    
    "two": number of two rupee coins,    
    "five": number of five rupee coins,    
    "ten": number of ten rupee coins,    
    "total": total value of the coins
  },  
  "image_en": base64 encoded string of the image with detected coins
}
```
