import socket, pyrah, struct, subprocess, multiprocessing, time, asyncio, serial_asyncio

# Shared Constants
DWP00 = bytes.fromhex("ff ff ff ff 00 00 00 00 00 00 00 00")
BUFFER_SIZE = 3 * 1024 * 1024
PORT_MAIN, PORT_BITSTREAM, CONTROL_PORT = 8080, 8081, 9090
UART_PORT, UART_BAUD, UART_TIMEOUT, MAX_UART_BUFFER = "/dev/ttyUSB0", 230400, 10, 10 * 1024

# States
CONNECTING, READ_CLIENT, WRITE_FPGA, READ_UART, WRITE_CLIENT = range(5)

class UARTProtocol(asyncio.Protocol):
  def __init__(self, shared_buffer, lock):
    self.shared_buffer, self.lock = shared_buffer, lock
  def connection_made(self, transport):
    print("UART: Serial connection opened")
  def data_received(self, data):
    asyncio.create_task(self._handle_data(data))
  async def _handle_data(self, data):
    async with self.lock:
      self.shared_buffer.extend(data)
      if len(self.shared_buffer) > MAX_UART_BUFFER:
        self.shared_buffer[:] = self.shared_buffer[-MAX_UART_BUFFER:]
  def connection_lost(self, exc):
    print("UART: Serial connection lost")

def main_server():
  async def run_main_server():
    shared_buffer = bytearray()
    uart_lock = asyncio.Lock()
    loop = asyncio.get_running_loop()
    await serial_asyncio.create_serial_connection(
        loop,
        lambda: UARTProtocol(shared_buffer, uart_lock),
        UART_PORT,
        baudrate=UART_BAUD)
    print(f"UART: Listening on {UART_PORT} @ {UART_BAUD}")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', PORT_MAIN))
    server_socket.listen(1)
    while True:
      print(f"MainServer: Waiting on port {PORT_MAIN}...")
      client_socket, addr = server_socket.accept()
      print(f"MainServer: Client {addr} connected")
      state, app_id, read_client_data = READ_CLIENT, 1, b""
      try:
        while True:
          if state == READ_CLIENT:
            app_id_data = client_socket.recv(4)
            if not app_id_data: break
            app_id = struct.unpack('>I', app_id_data)[0]
            length = struct.unpack('>I', client_socket.recv(4))[0]
            data = b""
            while len(data) < length:
              chunk = client_socket.recv(min(length - len(data), BUFFER_SIZE))
              if not chunk: break
              data += chunk
            if len(data) < length: break
            read_client_data, state = data, WRITE_FPGA
          if state == WRITE_FPGA:
            pyrah.rah_write(app_id, read_client_data)
            state = READ_UART if read_client_data[
                -12:] == DWP00 else READ_CLIENT
          if state == READ_UART:
            size = struct.unpack('>I', client_socket.recv(4))[0]
            print(f"MainServer: Waiting for {size} UART bytes")
            uart_data, timeout = b"", time.time() + UART_TIMEOUT
            while len(uart_data) < size:
              await asyncio.sleep(0.001)
              async with uart_lock:
                if len(shared_buffer) >= size:
                  uart_data = bytes(shared_buffer[:size])
                  del shared_buffer[:size]
                  break
              if time.time() > timeout:
                raise TimeoutError("MainServer: UART read timeout")
            state = WRITE_CLIENT
          if state == WRITE_CLIENT:
            client_socket.send(struct.pack('>I', len(uart_data)) + uart_data)
            state = READ_CLIENT
      except Exception as e:
        print(f"MainServer: Error: {e}")
      finally:
        client_socket.close()
        print("MainServer: Connection closed")
  asyncio.run(run_main_server())

# Bitstream flashing server
def bitstream_server():
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  s.bind(('', PORT_BITSTREAM))
  s.listen(1)
  print(f"BitstreamServer: Listening on port {PORT_BITSTREAM}...")
  try:
    while True:
      c, addr = s.accept()
      print(f"Bitstream: Client {addr}")
      length = struct.unpack('>I', c.recv(4))[0]
      data = b""
      while len(data) < length:
        data += c.recv(min(length - len(data), BUFFER_SIZE))
      with open("bitstream.hex", "wb") as f:
        f.write(data)
      subprocess.run(["sudo", "bitman", "-f", "bitstream.hex"])
      c.send(struct.pack('>I', 7) + b"Flashed")
      c.close()
  except Exception as e:
    print(f"BitstreamServer: Error: {e}")
  finally:
    s.close()
    print("BitstreamServer: Shutdown")

def parent_server():
  def start_servers():
    p1 = multiprocessing.Process(target=main_server, name="MainServer")
    p2 = multiprocessing.Process(target=bitstream_server,name="BitstreamServer")
    p1.start(), p2.start()
    return p1, p2
  p1, p2 = start_servers()
  ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  ctrl_sock.bind(('', CONTROL_PORT))
  ctrl_sock.listen(1)
  print(f"ParentServer: Listening on {CONTROL_PORT}...")
  try:
    while True:
      c, _ = ctrl_sock.accept()
      if c.recv(16).strip() == b"reset":
        print("ParentServer: Reset signal received")
        p1.terminate(), p2.terminate()
        p1.join(), p2.join()
        p1, p2 = start_servers()
        c.send(b"OK")
      c.close()
  except Exception as e:
    print(f"ParentServer: Error: {e}")
  finally:
    p1.terminate(), p2.terminate()
    ctrl_sock.close()

if __name__ == "__main__":
  parent_server()
