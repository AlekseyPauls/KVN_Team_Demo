# KVN Team Demo

## Setup

We recommend you to use Ubuntu or MacOS

Steps for setup:
1. Install requirements:
   
   ```pip install -r requirements```
   
    You may need to rebuild the OpenCV via CMake, this is normal - follow the instructions in the error message
2. Load weight for YOLO from `https://drive.google.com/file/d/1Vn_9dylrqgGVPL8U-FfASqiWVnOBkuHs/view?usp=sharing`
   and put it to `yolo` directory
3. Place known faces into `data/faces`
4. Run `streamlit run app.py`
5. Good work!

## Face Recognition

Choose `Realtime Face Recognition` from left side menu. Click `start`

## Personal Protective Equipment detection

Choose `Realtime Personal Protective Equipment detection` from left side menu. Click `start`