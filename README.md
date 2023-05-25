# Physics-informed GeoDeepONet
This project is a Python package for solving partial differential equations on parameterised geometries.

It provides a framework for defining and solving PDEs using DeepONets, a type of neural network
that can learn the solution operator to PDEs.
In this framework, arbitrary PDES can be solved by a forward pass through a neural network on general domains $\Omega_\phi = \phi(\Omega)$ parameterised by diffeomorphisms $\phi: \Omega \to \mathbb{R}^d$.


## Installation
To install the package, run:

`pip install geodeeponet`

## Usage
To use the package, you can import the necessary modules and classes and define your PDE, boundary conditions, and collocation points. An example on how to use this package can be found in the `poisson.py` file.

## Contributing
Contributions are welcome! If you find a bug or have a feature request, please open an issue on GitHub. If you want to contribute code, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
