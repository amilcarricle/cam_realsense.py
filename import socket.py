import socket
import hashlib

def calculate_checksum(data):
    return hashlib.md5(data.encode()).hexdigest()

def start_server():
    server_address = ('127.0.0.1', 5052)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(server_address)

    print(f"Server listening on {server_address}")

    last_sequence_number = -1

    while True:
        data, address = sock.recvfrom(4096)  # Buffer size is 4096 bytes
        if data:
            message = data.decode().strip()
            try:
                sequence_number, data_str, received_checksum = message.split(':')
                sequence_number = int(sequence_number)
                calculated_checksum = calculate_checksum(data_str)

                if received_checksum == calculated_checksum:
                    if sequence_number > last_sequence_number:
                        last_sequence_number = sequence_number
                        print(f"Received valid data from {address}: {data_str}")
                    else:
                        print(f"Out of order or duplicate packet received from {address}: Sequence {sequence_number}")
                else:
                    print(f"Checksum mismatch from {address}: Sequence {sequence_number}")
            except ValueError:
                print(f"Malformed packet received from {address}: {message}")

if __name__ == "__main__":
    start_server()
"""
server_address = ("127.0.0.1", 5052)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(server_address)

received_sequences = set()
expected_sequence_number = 1

def calculateChecksum(data):
    return hashlib.md5(data.encode()).hexdigest()

def verifyChecksum(data, checksum):
    return calculateChecksum(data) == checksum

print("Server is listening on {}:{}".format(*server_address))

while True:
    try:
        data, address = sock.recvfrom(4096)
        message = data.decode().strip()
        parts = message.split(":")
        
        if len(parts) != 3:
            print("Received malformed message:", message)
            continue

        sequence_number, data_str, checksum = parts
        sequence_number = int(sequence_number)

        if verifyChecksum(data_str, checksum):
            print(f"Received sequence number: {sequence_number}")
            if sequence_number == expected_sequence_number:
                print(f"Sequence {sequence_number} is in order")
                expected_sequence_number += 1
            else:
                print(f"Sequence {sequence_number} is out of order. Expected {expected_sequence_number}")
            
            received_sequences.add(sequence_number)
            print(f"Total unique sequences received: {len(received_sequences)}")
        else:
            print(f"Checksum verification failed for sequence {sequence_number}")
    except Exception as e:
        print(f"An error occurred: {e}")
"""
