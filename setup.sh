# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:28:29 2021

@author: 91779
"""

mkdir -p ~/.streamlit/
echo "
[general]n
email = "your-email@domain.com"n
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truen
enableCORS=falsen
port = $PORTn
" > ~/.streamlit/config.toml