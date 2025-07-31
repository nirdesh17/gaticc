import socket, struct, subprocess, multiprocessing, time, asyncio, spidev, pyrah

# Shared Constants
DWP00 = bytes.fromhex("ff ff ff ff 00 00 00 00 00 00 00 00")
PORT_MAIN, PORT_BITSTREAM, CONTROL_PORT = 8080, 8081, 9090
BUFFER_SIZE, MAX_SPI_BUFFER = 3 * 1024 * 1024, 10 * 1024
SPI_BUS, SPI_DEV, SPI_SPEED_HZ = 1, 0, 12_000_000

# States for Main Server
CONNECTING, READ_CLIENT, WRITE_FPGA, READ_FPGA, WRITE_CLIENT = range(5)

def print_hex_bytes(data, label=""):
  print(f"{label}{' ' if label else ''}{' '.join(f'{b:02x}' for b in data[:min(50, len(data))])} (len={len(data)})")

class SPIListener:
  def __init__(self, shared_buffer, lock):
    self.shared_buffer, self.lock = shared_buffer, lock
    self.spi = spidev.SpiDev()
    self.spi.open(SPI_BUS, SPI_DEV)
    self.spi.mode, self.spi.max_speed_hz, self.spi.bits_per_word = 0, SPI_SPEED_HZ, 8
    self.rolling = bytearray()

  async def poll(self):
    while True:
      await asyncio.sleep(0.00000001)
      recv = self.spi.xfer2([0x00] * 32)
      # print(' '.join(f'{b:02X}' for b in recv))
      self.rolling.extend(recv)
      while True:
        packet, consumed = self.extract_dwp_packet(self.rolling)
        if packet:
          async with self.lock:
            self.shared_buffer.extend(packet)
            if len(self.shared_buffer) > MAX_SPI_BUFFER:
              self.shared_buffer[:] = self.shared_buffer[-MAX_SPI_BUFFER:]
          del self.rolling[:consumed]
        else: break

  def extract_dwp_packet(self, data):
    start = data.find(b'\xFF\xFF\xFF\xFF')
    while start != -1:
      if start + 8 > len(data): break
      size = int.from_bytes(data[start+4:start+8], 'big')
      end = start + 4 + 4 + 4 + size + 12
      if end <= len(data) and data[end-12:end] == DWP00:
        return data[start:end], end
      start = data.find(b'\xFF\xFF\xFF\xFF', start + 1)
    return None, None

def main_server():
  async def run():
    state, app_id, read_client_data = CONNECTING, 1, b""
    shared_buffer, spi_lock = bytearray(), asyncio.Lock()
    asyncio.create_task(SPIListener(shared_buffer, spi_lock).poll())
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
      server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      server_socket.bind(('', PORT_MAIN))
      try:
        while True:
          if state == CONNECTING:
            print(f"State: CONNECTING {state}")
            server_socket.listen(1)
            print(f"Listening on {PORT_MAIN}...")
            client_socket, addr = server_socket.accept()
            print(f"Client connected from {addr}")
            state = READ_CLIENT
          if state == READ_CLIENT:
            print(f"State: READ_CLIENT {state}")
            app_id_data = client_socket.recv(4)
            if not app_id_data: state = CONNECTING; continue
            app_id = struct.unpack('>I', app_id_data)[0]
            length_bytes = client_socket.recv(4)
            if not length_bytes: state = CONNECTING; continue
            length = struct.unpack('>I', length_bytes)[0]
            data = b""
            while len(data) < length:
              chunk = client_socket.recv(min(length - len(data), BUFFER_SIZE))
              if not chunk: state = CONNECTING; break
              data += chunk
            read_client_data = data
            if len(data) == length: state = WRITE_FPGA
          if state == WRITE_FPGA:
            pyrah.rah_write(app_id, read_client_data)
            state = READ_FPGA if read_client_data[-12:] == DWP00 else READ_CLIENT
          if state == READ_FPGA:
            print(f"State: READ_FPGA {state}")
            length_bytes = client_socket.recv(4)
            if not length_bytes: state = CONNECTING; continue
            length = struct.unpack('>I', length_bytes)[0]
            print(f"Waiting for {length} bytes")
            spi_data, timeout = b"", time.time() + 10000
            while len(spi_data) < length:
              await asyncio.sleep(0.00000001)
              async with spi_lock:
                if len(shared_buffer) >= length:
                  spi_data = bytes(shared_buffer[:length])
                  del shared_buffer[:length]
                  break
              if time.time() > timeout: state = CONNECTING; break
            if len(spi_data) == length: state = WRITE_CLIENT
          if state == WRITE_CLIENT:
            print(f"State: WRITE_CLIENT {state}")
            client_socket.send(struct.pack('>I', len(spi_data)))
            client_socket.send(spi_data)
            state = READ_CLIENT
      except Exception as e:
        print(f"MainServer Error: {e}")
      finally:
        client_socket.close()
        print("MainServer shutdown")
  asyncio.run(run())

def bitstream_server():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', PORT_BITSTREAM))
    s.listen(1)
    print(f"Bitstream server on {PORT_BITSTREAM}...")
    try:
      while True:
        c, addr = s.accept()
        print(f"Bitstream - Client from {addr}")
        length = struct.unpack('>I', c.recv(4))[0]
        print(f"Receiving {length} bytes")
        data = b""
        while len(data) < length:
          data += c.recv(min(length - len(data), BUFFER_SIZE))
        with open("bitstream.hex", "wb") as f:
          f.write(data)
        subprocess.run(["sudo", "bitman", "-f", "bitstream.hex"])
        c.send(struct.pack('>I', 7))
        c.send(b"Flashed")
        c.close()
    except Exception as e:
      print(f"Bitstream Error: {e}")

def parent_server():
  def start():
    p1 = multiprocessing.Process(target=main_server)
    p2 = multiprocessing.Process(target=bitstream_server)
    p1.start(); p2.start()
    return p1, p2
  p1, p2 = start()
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ctrl_sock:
    ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ctrl_sock.bind(('', CONTROL_PORT))
    ctrl_sock.listen(1)
    print(f"Parent listening on {CONTROL_PORT}...")
    try:
      while True:
        c, _ = ctrl_sock.accept()
        cmd = c.recv(16).strip()
        if cmd == b"reset":
          print("Reset signal received")
          p1.terminate(); p2.terminate(); p1.join(); p2.join()
          p1, p2 = start()
          c.send(b"OK")
        c.close()
    except Exception as e:
      print(f"Parent Error: {e}")
    finally:
      p1.terminate(); p2.terminate()

if __name__ == "__main__":
  parent_server()
