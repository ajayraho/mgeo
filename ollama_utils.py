import requests
import json
import time
import threading
import subprocess


def run():
    """
    This function runs the Ollama server in a background thread.
    """
    def run_ollama_serve():
        try:
            # Start Ollama server
            subprocess.Popen(["ollama", "serve"])
        except Exception as e:
            # If this fails, we at least see why
            print(f"Failed to start 'ollama serve': {e}")

    thread = threading.Thread(target=run_ollama_serve, daemon=True)
    thread.start()
    # Give the server some time to boot
    time.sleep(5)


def call_ollama(prompt: str, system: str = "", retries: int = 8) -> str:
    """
    This function sends a prompt to the Ollama API, with retries in case of failure.
    It returns only the cleaned response text (discarding the thinking part).
    
    Parameters:
    - prompt (str): The prompt to send to the Ollama model.
    - system (str): Optional system prompt.
    - retries (int): Number of retries in case of failure.
    
    Returns:
    - str: The cleaned text response from the model.
    """
    print("Attempting to generate response with Ollama...")

    # Keep track if we already tried to start the server in this call
    tried_start_server = False

    for idx in range(retries):
        print(f"Attempt {idx + 1}/{retries} to call Ollama...")

        try:
            # Send the POST request to Ollama API
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",  # Ollama API endpoint
                json={
                    "model": "gpt-oss",  # Use the locally pulled model (adjust if needed)
                    "prompt": prompt,
                    "system": system,
                    "stream": False,  # Disable streaming for simplicity
                    "max_tokens": 100,  # Adjust max tokens as needed
                    "temperature": 0.2,  # You can tweak temperature or other params
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                },
                timeout=120
            )

            response.raise_for_status()  # Raise an exception for bad status codes
            json_response = response.json()

            result = json_response.get("response", "").strip()

            print("Ollama generated the response successfully.")
            return result

        except requests.exceptions.ConnectionError as e:
            # This is the "connection refused" / server not up case
            print(f"Connection error while contacting Ollama: {e}")

            if not tried_start_server:
                print("Ollama server seems down. Attempting to start it...")
                tried_start_server = True
                run()              # Start the server once
                time.sleep(5)      # Extra wait before retrying
            else:
                # We've already tried starting the server; backoff
                sleep_time = 3 + idx ** 2
                print(f"Already tried starting server. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        except requests.exceptions.HTTPError as e:
            # HTTP errors (4xx/5xx) - the server is reachable but returned an error
            status = e.response.status_code if e.response is not None else "unknown"
            print(f"HTTP error from Ollama (status {status}): {e}")

            if status == 404:
                # 404 could mean wrong endpoint or model missing
                print("Got 404 from Ollama. Check model name or endpoint.")
                # You *could* decide to break here or just retry:
                sleep_time = 3 + idx ** 2
                time.sleep(sleep_time)
            else:
                sleep_time = 3 + idx ** 2
                time.sleep(sleep_time)

        except requests.exceptions.RequestException as e:
            # Any other requests-level issue (timeouts, etc.)
            print(f"Request exception while contacting Ollama: {e}")
            sleep_time = 3 + idx ** 2
            time.sleep(sleep_time)

        except Exception as e:
            # Handle unexpected errors
            print(f"An unexpected error occurred: {e}")
            sleep_time = 3 + idx ** 2
            time.sleep(sleep_time)

    print("All attempts failed. Returning an empty string or default response.")
    return ""
