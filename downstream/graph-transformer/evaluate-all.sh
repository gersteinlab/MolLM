export data_path='./datasets/pcq-pos'                # path to data
export save_path='./logs/L12'    # path to checkpoints, e.g., ./logs/L12

export layers=12                                     # set layers=18 for 18-layer model
export hidden_size=768                               # dimension of hidden layers
export ffn_size=768                                  # dimension of feed-forward layers
export num_head=32                                   # number of attention heads
export num_3d_bias_kernel=128                        # number of Gaussian Basis kernels
export batch_size=256                                # batch size for a single gpu
export dataset_name="PCQM4M-LSC-V2-3D"
export add_3d="true"
bash evaluate.sh

#data_path='./datasets/pcq-pos' save_path='./logs/L12' layers=12 hidden_size=768 ffn_size=768 num_head=32 num_3d_bias_kernel=128 batch_size=256 dataset_name="PCQM4M-LSC-V2-3D" add_3d="true"