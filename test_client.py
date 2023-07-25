import random
import socket
import time


class QCorrector:
    def __init__(self, server_address):
        self.sock = None
        self.server_address = server_address
        self.count_send = 0
        self.errors = []
        self.Bob_key = []
        self.Alice_key = []

    # Read keys from files
    def read_keys(self, path_bob, path_alice):
        with open(path_bob, "r") as f:
            self.Bob_key = [int(i) for i in f.read()]

        with open(path_alice, "r") as f:
            self.Alice_key = [int(i) for i in f.read()]

    # Compute the XOR of a list of bits
    @staticmethod
    def compute_xor(bits):
        xor_value = bits[0]
        for bit in bits[1:]:
            xor_value ^= bit
        return xor_value

    # Request sum from server
    def request_sum_from_server(self, start, end):
        self.count_send += 1
        message = f"sum,{start},{end}"
        self.sock.sendall(message.encode('utf-8'))

        data = self.sock.recv(1024)
        return int(data.decode('utf-8'))

    # Recursive search for errors
    def search_errors(self, key, start, end):
        key_xor = self.compute_xor(key[start:end])
        server_xor = self.request_sum_from_server(start, end)

        # If there are only two bits, and they do not match the server's bits
        if len(key[start:end]) == 2 and key_xor != server_xor:
            return start, end - 1

        if key_xor != server_xor:
            error = self.search_errors(key, start, start + (end - start) // 2)
            if error is not None:
                self.errors.append(error)
            error = self.search_errors(key, start + (end - start) // 2, end)
            if error is not None:
                self.errors.append(error)

    # Error correction procedure
    def correct_errors(self, block_size=32, exit_threshold=3, max_iterations=100):
        exit_count = 0
        for _ in range(max_iterations):
            self.errors = []

            # Partition the key and search for errors in each block
            for i in range(0, len(self.Bob_key), block_size):
                start = i
                end = min(i + block_size, len(self.Bob_key))
                self.search_errors(self.Bob_key, start, end)

            # Analyze found errors
            for i, error in enumerate(self.errors):
                if error is not None:
                    next_index = 0 if error[1] + 1 == len(self.Bob_key) else error[1] + 1

                    a_xor1 = self.compute_xor([self.Bob_key[error[0] - 1], self.Bob_key[error[0]]])
                    b_xor1 = self.request_sum_from_server(error[0] - 1, error[0])
                    xor1 = a_xor1 == b_xor1

                    a_xor2 = self.compute_xor([self.Bob_key[error[1]], self.Bob_key[next_index]])
                    b_xor2 = self.request_sum_from_server(error[1], next_index)
                    xor2 = a_xor2 == b_xor2

                    if xor1 != xor2:
                        error_index = error[1] if xor1 else error[0]

                        # Flip the erroneous bit in Bob's key
                        self.Bob_key[error_index] = 1 - self.Bob_key[error_index]

                        self.errors[i] = None

            # After all errors have been processed, filter out the None values
            self.errors = [error for error in self.errors if error is not None]

            # If no errors are found for a certain number of iterations, stop
            if not self.errors:
                exit_count += 1
                if exit_count >= exit_threshold:
                    break

            # Otherwise, shuffle the key and repeat
            self.shuffle_key()

    # Shuffle the key
    def shuffle_key(self):
        seed = random.random()
        random.seed(seed)
        random.shuffle(self.Bob_key)

        message = f"rand,{seed}"
        self.sock.sendall(message.encode('utf-8'))
        random.seed(time.time())

    # Start the correction process
    def run(self, path_bob, path_alice, block_size=32):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server_address)

        self.read_keys(path_bob, path_alice)
        self.correct_errors(block_size)

        # Compare the final keys and count the number of errors
        num_errors = sum(a != b for a, b in zip(self.Bob_key, self.Alice_key))

        print(f"Bob's key: {self.Bob_key[:20]}")
        print(f"Alice's key: {self.Alice_key[:20]}")
        print(f"Number of errors: {num_errors}")

        # Send a final message to the server and close the connection
        final_message = f"end,{len(self.Bob_key) - 1},{len(self.Bob_key) - 1 + block_size}"
        self.sock.sendall(final_message.encode('utf-8'))

        self.sock.close()


if __name__ == "__main__":
    corrector = QCorrector(('localhost', 9090))
    corrector.run(path_bob="BOB_KEY (1).txt", path_alice="ALICE_KEY (1).txt")
