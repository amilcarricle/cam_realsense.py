import hashlib
import socket

def calculate_checksum(data):
    return hashlib.md5(data.encode()).hexdigest()

def send_udp_message(sock, server_address, sequence_number, data):
    data_str = ','.join(map(str, data))
    checksum = calculate_checksum(data_str)
    message = f"{sequence_number}:{data_str}:{checksum}\n"
    sock.sendto(message.encode(), server_address)

# Ejemplo de uso
server_address = ("127.0.0.1", 5052)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sequence_number = 1
data = [250, 257, 0.71, 269, 283, 0.71]  # Ejemplo de datos
send_udp_message(sock, server_address, sequence_number, data)