#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from app import app

if __name__ == '__main__':
    print("Starting Blood Group Prediction App...")
    print("Access the app at: http://127.0.0.1:5000/")
    app.run(debug=False, use_reloader=False)
