import os
import socket
import serial
import json
from pymongo import MongoClient
from datetime import datetime
import time
import threading

# Configuration
ETHERNET_PORT = 5001
SERIAL_PORT = '/dev/ttyS0'
BAUD_RATE = 9600
OFFLINE_STORAGE = './offline_data'

# MongoDB Configuration
cluster = MongoClient("mongodb+srv://ben10promax:ben10promax@ben10promax.giwzamd.mongodb.net/?retryWrites=true&w=majority&appName=ben10promax")
db = cluster["violations"]
coll = db["datas"]

# Ensure offline storage directory exists
os.makedirs(OFFLINE_STORAGE, exist_ok=True)

# Initialize serial communication with Arduino
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

def is_internet_available():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False

def save_offline(data):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = os.path.join(OFFLINE_STORAGE, f'{timestamp}.json')
    with open(filename, 'w') as f:
        json.dump(data, f)

def send_to_mongodb(data):
    try:
        # Find the highest existing _id in the collection
        last_post = coll.find_one(sort=[("_id", -1)])  # Sort by _id in descending order
        new_id = (last_post["_id"] + 1) if last_post else 1  # Increment _id or start at 1 if no documents exist

        # Add the incrementing ID, timestamp, license_plate, and vehicle_type to the data
        data["_id"] = new_id
        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        data["license_plate"] = data.get("license_plate", "Unknown")
        data["vehicle_type"] = data.get("vehicle_type", "Unknown")

        # Insert the document into the collection
        coll.insert_one(data)
        print("Data sent to MongoDB with _id:", new_id)
    except Exception as e:
        print("Failed to send data to MongoDB:", e)
        save_offline(data)

def process_file(data):
    try:
        # Extract license_plate and vehicle_type from the JSON data
        license_plate = data.get('license_plate')
        vehicle_type = data.get('vehicle_type')
        image_filename = data.get('image_filename')

        if not license_plate or not vehicle_type:
            print("Invalid data received")
            return

        # Log the received data
        print(f"License Plate: {license_plate}, Vehicle Type: {vehicle_type}, Image: {image_filename}")

        # Send data to Arduino
        arduino_data = f"{license_plate},{vehicle_type}\n"
        arduino.write(arduino_data.encode())
        print("Data sent to Arduino:", arduino_data)

        # Check internet connectivity
        if is_internet_available():
            # Send current data to MongoDB
            send_to_mongodb(data)

            # Send offline data to MongoDB
            for offline_file in os.listdir(OFFLINE_STORAGE):
                offline_path = os.path.join(OFFLINE_STORAGE, offline_file)
                with open(offline_path, 'r') as f:
                    offline_data = json.load(f)
                    send_to_mongodb(offline_data)
                os.remove(offline_path)
        else:
            # Save data locally if no internet
            save_offline(data)

    except Exception as e:
        print("Error processing file:", e)

def receive_files(client_socket):
    """Receive multiple files (JSON and picture) sent over Ethernet."""
    try:
        while True:
            # Receive file metadata
            metadata = client_socket.recv(1024).decode()
            if not metadata:
                break
            filename, filesize = metadata.split("<SEPARATOR>")
            filesize = int(filesize)

            # Save the received file
            filepath = os.path.join(OFFLINE_STORAGE, filename)
            with open(filepath, "wb") as f:
                bytes_received = 0
                while bytes_received < filesize:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_received += len(chunk)
            print(f"Received file: {filename}")

            # Process JSON file if received
            if filename.endswith(".json"):
                with open(filepath, "r") as json_file:
                    data = json.load(json_file)
                    process_file(data)

    except Exception as e:
        print(f"Error receiving files: {e}")

def main():
    # Set up socket for listening
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('', ETHERNET_PORT))
    server_socket.listen(5)
    print(f"Listening on port {ETHERNET_PORT}...")

    while True:
        try:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")

            # Receive files (JSON and picture)
            receive_files(client_socket)

            client_socket.close()
        except Exception as e:
            print("Error in main loop:", e)

if __name__ == '__main__':
    # Start a thread for the main server loop
    server_thread = threading.Thread(target=main, daemon=True)
    server_thread.start()

    # Keep the main thread alive
    while True:
        time.sleep(1)