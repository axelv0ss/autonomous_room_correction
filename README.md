# autonomous_room_correction

## Project Motivation
Digital room correction aims to actively compensate for acoustic distortions caused by the room and speakers by applying processing to the source signal. This is traditionally accomplished in three steps. First, the acoustic properties of the room-speaker system are estimated. Secondly, the correction parameters are calculated based on these estimates. Lastly, the correction filters are applied in real-time.

The objectives of room correction include flattening the frequency response, reducing reflections and reverberation and obtaining spatially uniform acoustics. In practice, these objectives are conflicting, making it challenging to attain an acoustically transparent room- speaker system. Current methods require a time-consuming calibration process and fail to adjust when the acoustics environment changes, such as when a crowd enters the room. This project aims to develop an autonomous room correction algorithm that can be deployed in both home and live situations.

## Evolutionary Computation
Evolutionary computation is used as a means to explore the solution space. The algorithm has the following general form:
1. Generate an initial population of different filter configurations
2. Determine their relative performances by measuring their corresponding RTFs
3. Generate a new population by cross-breeding parameters from the top performers
4. Return to step 2
