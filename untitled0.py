# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:54:57 2021

@author: Elijah_Nkuah
"""
from webcam_component import webcam

captured_image = webcam()
if captured_image is not None:
   st.image(captured_image)