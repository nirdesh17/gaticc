import gati

def gen():
    arr = np.load("mnist_10.npy")
    return arr

def post(num):
    m = np.argmax(num)
    with open("results.txt", "a") as f: f.write(f"{m}\n")
    print(f"number: {m}")
    return m

if __name__ == "__main__":
    onnx_path = "/home/metal/dev/gaticc/tests/models/mnist_average_pool_int8.onnx"
    with open("results.txt", "w"): pass
    gati.sim(onnx_path, "mnist.py", "gen", "post")
    print(f"Match: {gati.match('mnist_10_labels.txt', 'results.txt')}%")
