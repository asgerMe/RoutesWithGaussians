import numpy as np
import os
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app

class GetData:

    def __init__(self, API_KEY):
        self.api_key = API_KEY
        cred = credentials.Certificate('key.json')
        default_app = initialize_app(cred)
        db = firestore.client()

    def __repr__(self):
        return 'Queue new data points and fetch proxy data'

    def __add__(self, other):
        return 0

    def __sub__(self, other):
        return 0

