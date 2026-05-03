**Dataset**
- num_samples: 100
- x_range: (-2, 2)
- target_function: sin(10x)

**Model**
- num_hidden_nodes: 60
- mode: fourier (options: raw, linear, fourier)
- num_features: 50
- scale: 10

**Architecture**
- input_dim: depends on mode
- hidden_layer: Linear(input_dim → 60)
- activation: tanh
- output_layer: Linear(60 → 1)

**Fourier Features**
- num_features: 50
- output_dim: 100 (sin + cos)
- scale: 10

**Linear Features**
- num_features: 50
- scale: 1.0

**Training**
- loss: MSELoss
- optimizer: Adam
- learning_rate: 0.001
- epochs: 1000
- batch_size: 16
- shuffle: True

**Evaluation**
- plot_range: (-2, 2)
- num_points: 200
