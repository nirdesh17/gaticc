# Simulator for CNN

## Dependencies for this Simultor :


- Fedora
    
``` $ sudo dnf install boost-devel python3-devel protobuf-compiler ```



---

## GOALS:
    1. existing code clean : adding comments ( tree, already marked explain further)
    2. max_pooler proper plugable (class) flexible
    3. quantization 
    4. batchNorm
    5. bias blocks
    6. representation/execution of a neural network(graph) onnx****
    7. hybrid SA of SA   SASA
    8. type strictness (reg_t)
    9. memory leaks (valgrind ) 
    10. const correctness
    11. static analysis(coverity)
    12. fuzzy testing 
    12. LIT integreation 
    13. Mat template for floats
    14. compiler gati-cc
    15. multi size kernels
    16. strided convo
    17. DW/PW convo
    18. dilated convo
    19. padded convo


### Questions: 
To what extent we need to sim

    * max_pool 
    * output block(addr trees)
    * xbars/ FIFO
