import os
import requests

# variable d'api
api_adress = '0.0.0.0'
api_port = 5000
api_endpoint = "metrics/LR"
# variable de test
param_user = "Quinlan"
param_psw = 5210
expect_result = "200"
param_meteo= "rainy"
param_temp = 10
param_bike= 327
param_day="Wednesday"


r = requests.post(url='http://{}:{}/{}'.format(api_adress,api_port,api_endpoint), json={'username':'{}'.format(param_user),'password': param_psw, 'meteo':'{}'.format(param_meteo), 'temp': param_temp, 'bike': param_bike,'day':'{}'.format(param_day)})

output = '''
==================================
	test metrics_LR
==================================

request done at "/{endpoint}"
| username = {user}
| password = {psw}
| meteo = {meteo}
| temperature = {temp}
| number of bikes = {bike}
| jour de la semaine = {day}


=> {test}
'''

status_code = r.status_code
if status_code == int(expect_result):
	test_result = r.text
else: test_result = 'FAILURE'
print(output.format(endpoint=api_endpoint,user=param_user,psw=param_psw,meteo=param_meteo,temp=param_temp,bike=param_bike,day=param_day,test=test_result))


