# write the API for the MOXA
from time import sleep
import requests
import logging

"""
Controller for the MOXA: Connection for FESTO Ring Light
"""

class MoxaRelay:
    def __init__(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        self.ip = '192.168.127.254'


    def getRelayInformation(self, relayNum):
        url = f'http://{self.ip}/api/slot/0/io/relay/{relayNum}/relayStatus'
        return self.__get(url)


    def setLightOn(self):
        url = f'http://{self.ip}/api/slot/0/io/relay/2/relayStatus'
        data_ON = {"slot":"0","io":{"relay":{"2":{"relayStatus":"1"}}}}
        self.__post(url, data_ON)




    def setLightOff(self):
        url = f'http://{self.ip}/api/slot/0/io/relay/2/relayStatus'
        data_OFF = {"slot":"0","io":{"relay":{"2":{"relayStatus":"0"}}}}
        self.__post(url, data_OFF)


    def __get(self, url):
        headers = {
            'Accept': 'vdn.dac.v1',
            'Content-Type': 'application/json'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            logging.info("Get request sent successfully:")
    
        else:
            logging.error("Get request failed with status code:", response.status_code)


    def __post(self, url, data):
        headers = {
            'Accept': 'vdn.dac.v1',
            'Content-Type': 'application/json'
        }
        response = requests.put(url, json=data, headers=headers)
        if response.status_code == 200:
            logging.info("Post sent successfully.")
        else:
            logging.error("Post request failed with status code:", response.status_code)



if __name__ == "__main__":
    moxa = MoxaRelay()
    moxa.setLightOn()
    # sleep(2)
    # moxa.setLightOff()

