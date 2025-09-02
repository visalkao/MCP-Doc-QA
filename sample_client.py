import requests

URL = 'http://localhost:8080'

with open('Internship Report - Visal KAO e20190354.pdf','rb') as f:
    r = requests.post(URL + '/upload', files={'file': ('sample.pdf', f, 'application/pdf')})
    print(r.json())

r = requests.get(URL + '/query', params={'q': 'What is the invoice number on the first page?'})
print(r.json())
