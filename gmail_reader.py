import os
import base64
import requests
import sqlite3

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

API_URL = "http://127.0.0.1:8000/analyze"


# --------------------------------------------------
# GMAIL AUTHENTICATION
# --------------------------------------------------

def authenticate_gmail():

    creds = None

    # Load previously saved credentials
    if os.path.exists("token.json"):

        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    # Authenticate if credentials are missing/invalid
    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:

            creds.refresh(Request())

        else:

            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        # Save credentials for future runs
        with open("token.json", "w") as token:

            token.write(creds.to_json())

    return creds


# --------------------------------------------------
# EXTRACT EMAIL BODY
# --------------------------------------------------

def get_email_body(payload):

    # Email has a direct body
    if payload.get("body", {}).get("data"):

        return base64.urlsafe_b64decode(
            payload["body"]["data"]
        ).decode(
            "utf-8",
            errors="ignore"
        )

    # Multipart email
    for part in payload.get("parts", []):

        # Prefer plain text
        if part.get("mimeType") == "text/plain":

            data = part.get("body", {}).get("data")

            if data:

                return base64.urlsafe_b64decode(
                    data
                ).decode(
                    "utf-8",
                    errors="ignore"
                )

        # Handle nested multipart sections
        if part.get("parts"):

            body = get_email_body(part)

            if body:
                return body

    return ""


# --------------------------------------------------
# GET EMAIL HEADER
# --------------------------------------------------

def get_header(headers, name):

    for header in headers:

        if header["name"].lower() == name.lower():

            return header["value"]

    return ""


# --------------------------------------------------
# CHECK WHETHER EMAIL WAS ALREADY PROCESSED
# --------------------------------------------------

def get_processed_message_ids():

    conn = sqlite3.connect("support_cases.db")

    cursor = conn.cursor()

    rows = cursor.execute(
        """
        SELECT message_id
        FROM cases
        WHERE message_id IS NOT NULL
        """
    ).fetchall()

    conn.close()

    return {
        row[0]
        for row in rows
    }


# --------------------------------------------------
# READ LATEST NEW EMAIL
# --------------------------------------------------

def read_latest_email():

    # Authenticate with Gmail
    creds = authenticate_gmail()

    # Create Gmail API service
    service = build(
        "gmail",
        "v1",
        credentials=creds
    )

    # ----------------------------------------------
    # FIND UNREAD EMAILS
    # ----------------------------------------------

    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        q="is:unread",
        maxResults=10
    ).execute()

    messages = results.get("messages", [])

    if not messages:

        print("No unread emails found.")

        return


    # ----------------------------------------------
    # CHECK FOR DUPLICATES
    # ----------------------------------------------

    processed_ids = get_processed_message_ids()

    message_id = None

    for msg in messages:

        if msg["id"] not in processed_ids:

            message_id = msg["id"]

            break


    if message_id is None:

        print("No new unread emails to process.")

        return


    # ----------------------------------------------
    # GET COMPLETE EMAIL
    # ----------------------------------------------

    message = service.users().messages().get(
        userId="me",
        id=message_id,
        format="full"
    ).execute()


    payload = message["payload"]

    headers = payload.get("headers", [])


    # ----------------------------------------------
    # EXTRACT EMAIL INFORMATION
    # ----------------------------------------------

    sender = get_header(
        headers,
        "From"
    )

    subject = get_header(
        headers,
        "Subject"
    )

    received_at = get_header(
        headers,
        "Date"
    )

    body = get_email_body(payload)


    # ----------------------------------------------
    # DISPLAY EMAIL
    # ----------------------------------------------

    print("\n========== EMAIL FOUND ==========")

    print("Message ID:")
    print(message_id)

    print("\nFrom:")
    print(sender)

    print("\nSubject:")
    print(subject)

    print("\nBody:")
    print(body)

    print("\n=================================\n")


    # ----------------------------------------------
    # SEND EMAIL TO FASTAPI
    # ----------------------------------------------

    try:

        response = requests.post(
            API_URL,
            params={
                "subject": subject,
                "message_body": body,
                "sender": sender,
                "message_id": message_id,
                "received_at": received_at
            },
            timeout=30
        )


        print("API response:")
        print(response.status_code)


        # If FastAPI returned JSON
        try:

            print(response.json())

        except ValueError:

            print("API returned non-JSON response:")
            print(response.text)


    except requests.exceptions.ConnectionError:

        print(
            "\nERROR: Could not connect to FastAPI."
        )

        print(
            "Make sure this is running:"
        )

        print(
            "uvicorn app:app --reload"
        )


    except requests.exceptions.Timeout:

        print(
            "\nERROR: FastAPI request timed out."
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    read_latest_email()