import socket, struct, subprocess, multiprocessing, time, asyncio, spidev, pyrah, argparse

# Shared Constants
DWP00 = bytes.fromhex("ff ff ff ff 00 00 00 00 00 00 00 00")
PORT_MAIN, PORT_BITSTREAM, CONTROL_PORT = 8080, 8081, 9090
BUFFER_SIZE, SPI_READ_TIMEOUT_S = 3 * 1024 * 1024, 6
SPI_BUS, SPI_DEV, SPI_SPEED_HZ = 1, 0, 8_000_000

# States for Main Server
CONNECTING, READ_CLIENT, WRITE_FPGA, READ_FPGA, WRITE_CLIENT = range(5)

# Dispatch
multi_dispatch, verbose, verbose2 = False, False, False
total_num_layers = 0

# sleep time in seconds
_sleep_time = 0

log  = lambda m: verbose  and print(m)
log2 = lambda m: verbose2 and print(m)

class SPIListener:
    def __init__(self):
        self.spi = spidev.SpiDev(); self.spi.open(SPI_BUS, SPI_DEV)
        self.spi.mode = 0; self.spi.max_speed_hz = SPI_SPEED_HZ
        self.spi.bits_per_word = 8; self.spi_buffer_size = 16 
        self.rolling = bytearray()

    def extract_dwp_packet(self, data: bytearray):
        start = data.find(b'\xFF\xFF\xFF\xFF')
        while start != -1:
            if start + 12 > len(data): break
            size = int.from_bytes(data[start + 4:start + 8], 'big')
            end = start + 4 + 4 + 4 + size + 12
            if end > len(data): break
            if data[end - 12:end] == DWP00:
                return bytes(data[start:end]), end
            start = data.find(b'\xFF\xFF\xFF\xFF', start + 1)
        return None, None

    async def read_exact_dwp(self, expected_payload_size):
        start_time = time.monotonic()
        while True:
            await asyncio.sleep(_sleep_time)
            recv = self.spi.xfer2([0x00] * self.spi_buffer_size)
            log2(' '.join(f'{b:02X}' for b in recv))
            self.rolling.extend(recv)

            if all(b==0x00 for b in recv) and time.monotonic() - start_time > SPI_READ_TIMEOUT_S:
                print("Timeout, got all zeroes"); return None
            start = self.rolling.find(b'\xFF\xFF\xFF\xFF')
            if start != -1: self.rolling = self.rolling[start:]; break

        while len(self.rolling) < 8:
            await asyncio.sleep(_sleep_time)
            recv = self.spi.xfer2([0x00] * self.spi_buffer_size)
            log2(' '.join(f'{b:02X}' for b in recv))
            self.rolling.extend(recv)

        payload_size = int.from_bytes(self.rolling[4:8], 'big')
        total_size = 4 + 4 + 4 + payload_size + 12
        log(f"Got payload size from FPGA = {payload_size}")
        log(f"Got Total Bytes from FPGA = {total_size}")

        bytes_needed = total_size - len(self.rolling)
        reads_needed = -(-bytes_needed // self.spi_buffer_size)
        # reads_needed = 100000
        log(f"Number of spi reads needed = {reads_needed}")
        for i in range(reads_needed):
            await asyncio.sleep(_sleep_time)
            recv = self.spi.xfer2([0x00] * self.spi_buffer_size)
            log2(f' '.join(f'{b:02X}' for b in recv))
            self.rolling.extend(recv)

        dwp_packet, consumed = self.extract_dwp_packet(self.rolling)
        if dwp_packet and len(dwp_packet) == total_size:
            del self.rolling[:consumed]
            return dwp_packet
        else:
            print("!! Failed to parse full packet after exact reads !!")
            return None

def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

def main_server():
    log(f"Running @ {SPI_SPEED_HZ} Hz!")

    async def run():
        global multi_dispatch
        global total_num_layers
        state, app_id, read_client_data = CONNECTING, 1, b""
        spi_listener = SPIListener()
        layers_remaining = 0  # NEW: track per-connection layers

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('', PORT_MAIN))

            while True:
                if state == CONNECTING:
                    print(f"State: CONNECTING {state}")
                    server_socket.listen(1)
                    client_socket, addr = server_socket.accept()
                    print(f"Client connected from {addr}")
                    try:
                        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except Exception:
                        pass
                    state = READ_CLIENT

                    num_layers_bytes = recv_exact(client_socket, 4)
                    if not num_layers_bytes:
                        print("Client disconnected before sending number of layers")
                        state = CONNECTING
                        continue
                    total_num_layers = struct.unpack('>I', num_layers_bytes)[0]
                    layers_remaining = total_num_layers
                    log(f"num_layers_to_dispatch = {layers_remaining}")
                    
                    multi_dispatch = True if total_num_layers>1 else False
                    log(f"multi_dispatch set to {multi_dispatch}")

                if state == READ_CLIENT:
                    print(f"State: READ_CLIENT {state}")
                    header = client_socket.recv(4)
                    if not header:
                        print("Client disconnected")
                        state = CONNECTING
                        continue

                    app_id = struct.unpack('>I', header)[0]
                    client_socket.setblocking(False)
                    try:
                        peek_bytes = client_socket.recv(4, socket.MSG_PEEK)
                    except BlockingIOError:
                        peek_bytes = b''
                    client_socket.setblocking(True)

                    length_bytes = client_socket.recv(4)
                    if not length_bytes:
                        print("Client disconnected")
                        state = CONNECTING
                        continue

                    length = struct.unpack('>I', length_bytes)[0]
                    data = recv_exact(client_socket, length)
                    if not data:
                        print("Connection lost during READ_CLIENT")
                        state = CONNECTING
                        continue

                    read_client_data = data
                    state = WRITE_FPGA

                if state == WRITE_FPGA:
                    pyrah.rah_write(app_id, read_client_data)
                    if read_client_data[-12:] == DWP00:
                        state = READ_FPGA
                    else:
                        state = READ_CLIENT

                if state == READ_FPGA:
                    length_bytes = recv_exact(client_socket, 4)
                    log2(f"Length to read : {length_bytes}")
                    if not length_bytes:
                        print("Client disconnected while waiting for read length")
                        state = CONNECTING
                        continue
                    payload_len = struct.unpack('>I', length_bytes)[0]
                    print(f"State: READ_FPGA {state}")
                    log(f"Expected bytes {payload_len}")
                    spi_data = await spi_listener.read_exact_dwp(payload_len)
                    if not spi_data:
                        print("No data coming from FPGA..")
                        state = CONNECTING
                        continue
                    state = WRITE_CLIENT

                if state == WRITE_CLIENT:
                    client_socket.send(struct.pack('>I', len(spi_data)))
                    client_socket.send(spi_data)

                    if multi_dispatch:
                        layers_remaining -= 1
                        if layers_remaining > 0:
                            log(f"layers_remaining = {layers_remaining}")
                            state = READ_FPGA
                        else:
                            layers_remaining = total_num_layers
                            state = READ_CLIENT
                    else:
                        state = READ_CLIENT
    asyncio.run(run())

def bitstream_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', PORT_BITSTREAM))
        s.listen(1)
        print(f"Bitstream server on {PORT_BITSTREAM}...")
        while True:
            c, addr = s.accept()
            print(f"Bitstream - Client from {addr}")
            length = struct.unpack('>I', c.recv(4))[0]
            data = recv_exact(c, length)
            with open("bitstream.hex", "wb") as f:
                f.write(data)
            subprocess.run(["sudo", "bitman", "-f", "bitstream.hex"])
            c.send(struct.pack('>I', 7))
            c.send(b"Flashed")
            c.close()


def parent_server():
    def start():
        p1 = multiprocessing.Process(target=main_server)
        p2 = multiprocessing.Process(target=bitstream_server)
        p1.start()
        p2.start()
        return p1, p2

    p1, p2 = start()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ctrl_sock:
        ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ctrl_sock.bind(('', CONTROL_PORT))
        ctrl_sock.listen(1)
        print(f"Parent listening on {CONTROL_PORT}...")
        while True:
            c, _ = ctrl_sock.accept()
            cmd = c.recv(16).strip()
            if cmd == b"reset":
                print("Reset signal received")
                p1.terminate()
                p2.terminate()
                p1.join()
                p2.join()
                p1, p2 = start()
                c.send(b"OK")
            c.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPI server with optional verbose logging.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug output.")
    parser.add_argument("-v2", "--verbose2", action="store_true", help="Enable verbose2 for more debug output.")
    args = parser.parse_args()
    verbose, verbose2 = args.verbose, args.verbose2
    parent_server()
