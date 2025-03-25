import gati

if __name__ == "__main__":
    onnx_path = "../../tests/models/vgg_16_224_int8.onnx"
    bitstream = "../../rah_gati_0.4.3.hex"
    gml_path = "model.gml"
    gati.set_remote("192.168.0.130")
    gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
    #gati.compile(onnx_path, gml_path)
    gati.flash(bitstream)
    with open("results.txt", "w"): pass
    ret = gati.run(onnx_path, gml_path, "imagenet.py", "gen", "post", "v")
    if not ret:
        print(f"Match: {gati.match('imagenet_100_labels.txt', 'results.txt')}%")
