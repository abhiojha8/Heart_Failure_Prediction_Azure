import requests
import json

scoring_uri = 'http://d50ff359-fc30-43a0-9039-a698a6b690c2.eastus2.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'T5zinCIOlRIrERZ4lpndfKdbjVAalIbV'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
           "age": 61, 
           "anaemia": 0, 
           "creatinine_phosphokinase": 582, 
           "diabetes": 1, 
           "ejection_fraction": 20, 
           "high_blood_pressure": 1, 
           "platelets": 265000, 
           "serum_creatinine": 1.9, 
           "serum_sodium": 130, 
           "sex": 1, 
           "smoking": 0,
           "time": 4
          }
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())