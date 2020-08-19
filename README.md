# backSub4Lectures
The program I use to perform background subtraction to include my face over the slides in the Open Broadcaster Software (OBS). 

Using Chroma key in OBS, and a bed sheet as a screen behind me, did not do the job. I use OpenCV to do some background subtraction and replace the background with a uniform colour, e.g. green. 

- run the tool, e.g. with ./backSub4Lectures/.py in Linux. Make sure OpenCV and Numpy are installed for Python 3.
- Click-n-drag to select a region of interest (ROI)
- Hide under your desk and press 'b' to initialiase the background.
- Adjust the slider if needed.
- Keep the program running.
- Launch OBS
- Add the Window as video input
- Add chroma key filter
- here you go
