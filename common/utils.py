import firebase_admin
from firebase_admin import credentials, firestore
import os
from common.config import FIREBASE_KEY_JSON_PATH

def initFirestore():
    cred_path = os.path.join(os.path.dirname(__file__), FIREBASE_KEY_JSON_PATH)
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()
