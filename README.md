# Autonomous Room Correction

## Dependencies
* macOS operating system
* Python 3.6 or later
* Anaconda
* Soundflower for audio routing

## Run
* Configure the params.py to reflect the input and output stream indices of your system
* Run main.py

## Project Motivation
Digital room correction aims to actively compensate for acoustic distortions caused by the room and speakers by applying processing to the source signal. This is traditionally accomplished in three steps. First, the acoustic properties of the room-speaker system are estimated. Secondly, the correction parameters are calculated based on these estimates. Lastly, the correction filters are applied in real-time. This project aims to develop an autonomous room correction algorithm that can be deployed in both home and live situations.
