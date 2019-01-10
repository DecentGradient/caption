import requests
import json
import  base64
url = 'http://localhost:8501/v1/models/caption:predict'
headers = { 'Content-Type':'application/json'}
with open("./toinfer/giraffe.jpg","rb")as pic:
    enc = pic.read()
    enc = base64.b64encode(enc)
    enc = enc.decode("utf-8")
    # instance=  [{"image_bytes": {"b64": enc}}]
    # instance=  [{"image_bytes": {"b64": enc}}]
    instance = [{"b64":enc}]
    data = json.dumps({"instances":instance})
    print(data[0:100])
res = requests.post(url,data=data)

print(res.text)
print(res)