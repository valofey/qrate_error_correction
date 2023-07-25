import socket
import random
import time
import hashlib


class Server:
    def __init__(self, address, key_file, block_size=4, matrix_height=3, matrix_length=7):
        self.server_address = address
        self.block_size = block_size
        self.matrix_height = matrix_height
        self.matrix_length = matrix_length
        self.matrix = [[1, 1, 1, 0, 1, 0, 0], [0, 1, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1]]

        # Load Alice's key from file
        with open(key_file, "r") as f:
            self.Alice_key = [int(i) for i in f.read()]

    # Handle a single client request
    def handle_request(self, conn):
        data = conn.recv(1024)
        message = data.decode('utf-8')
        command = message.split(",")

        if command[0] == "rand":
            self.shuffle_key(float(command[1]))
            conn.send(self.encode_key().encode('utf-8'))

        elif command[0] == "srav":
            conn.send(str(self.Alice_key[int(command[1])]).encode('utf-8'))

        elif command[0] == "hesh":
            start = int(command[1])
            end = int(command[2])
            hashed_str = hashlib.md5(str(self.Alice_key[start:end]).encode()).hexdigest()
            conn.send(hashed_str.encode('utf-8'))

        elif command[0] == "del":
            for j in range(int(command[1]), int(command[2]) + 1):
                self.Alice_key[j] = -1

        elif command[0] == "endhash":
            self.Alice_key = [bit for bit in self.Alice_key if bit != -1]

        elif command[0] == "amplifier":
            # TODO: Implement key amplification if needed
            pass

        else:
            print("Unknown command")

    # Shuffle the key
    def shuffle_key(self, seed):
        random.seed(seed)
        random.shuffle(self.Alice_key)
        random.seed(time.time())

    # Encode the key using matrix multiplication
    def encode_key(self):
        encoded_key = ""
        for i in range(0, len(self.Alice_key), self.block_size):
            for q in range(self.matrix_height):
                encoded_bit = self.Alice_key[i] * self.matrix[q][0]
                for j in range(1, self.block_size):
                    encoded_bit ^= self.Alice_key[i + j] * self.matrix[q][j]
                encoded_key += str(encoded_bit)
        return encoded_key

    # Start the server
    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(self.server_address)
            s.listen(1)

            while True:
                conn, addr = s.accept()
                with conn:
                    print('Connected by', addr)
                    self.handle_request(conn)


if __name__ == "__main__":
    server = Server(('localhost', 9090), "ALICE_KEY (2).txt")
    server.run()
