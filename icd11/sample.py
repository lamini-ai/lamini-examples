import requests

token_endpoint = 'https://icdaccessmanagement.who.int/connect/token'
client_id = 'cf8ad7e4-8d98-4064-86f0-c32e98197fa9_73cbc95d-f750-49b2-96ca-37ea9cc2bdc3'
client_secret = 'U3VF2q06kJW00OVTLPLjBvDD4hchxAUYPlB1qQhWL64='
scope = 'icdapi_access'
grant_type = 'client_credentials'


# get the OAUTH2 token

# set data to post
payload = {'client_id': client_id,
	   	   'client_secret': client_secret,
           'scope': scope,
           'grant_type': grant_type}

# make request
r = requests.post(token_endpoint, data=payload, verify=False).json()
token = r['access_token']


# access ICD API

# uri = 'https://id.who.int/icd/entity'
# uri = 'https://id.who.int/icd/entity/448895267'
uri = 'https://id.who.int/icd/release/11/2023-01/mms'
# uri='http://id.who.int/icd/release/11/2023-01/mms/1766440644'
# uri='http://id.who.int/icd/release/11/2023-01/mms/809926550'
# uri = 'http://id.who.int/icd/release/11/2023-01/mms/135898846'
# uri = 'http://id.who.int/icd/release/11/2023-01/mms/1296093776'
# uri = 'http://id.who.int/icd/release/11/2023-01/mms/2091156678'
# uri='http://id.who.int/icd/release/11/2023-01/mms/1611724421'
# uri = 'http://id.who.int/icd/release/11/2023-01/mms/223744320'
# uri = 'http://id.who.int/icd/release/11/2023-01/mms/775270311'
# uri = 'http://id.who.int/icd/release/11/2023-01/mms/393773440'
# uri = 'http://id.who.int/icd/release/11/2023-01/mms/186534168/other'

# HTTP header fields to set
headers = {'Authorization':  'Bearer '+token,
           'Accept': 'application/json',
           'Accept-Language': 'en',
	   'API-Version': 'v2'}

# make request
r = requests.get(uri, headers=headers, verify=False)

# print the result
print (r.text)
